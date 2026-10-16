"""Public Kaggle downloads and training-only tabular preprocessing.

No credentials are read. A successful first download records the observed
version/license and hashes; those hashes detect later local cache changes.
They are not a publisher-supplied authenticity signature.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile

CATALOG = {
    "breast_cancer": dict(
        ref="uciml/breast-cancer-wisconsin-data", file="data.csv", task="classification",
        target="diagnosis", labels=("B", "M"), categorical=(),
        numeric=tuple(f"{name}_{suffix}" for suffix in ("mean", "se", "worst") for name in (
            "radius", "texture", "perimeter", "area", "smoothness", "compactness",
            "concavity", "concave points", "symmetry", "fractal_dimension")),
        purpose="Correlated numerical measurements with different units; scaling and curvature."),
    "adult": dict(
        ref="uciml/adult-census-income", file="adult.csv", task="classification",
        target="income", labels=("<=50K", ">50K"),
        numeric=("age", "education.num", "capital.gain", "capital.loss", "hours.per.week"),
        categorical=("workclass", "education", "marital.status", "occupation", "relationship",
                     "race", "sex", "native.country"),
        purpose="Imbalanced income classification; one-hot CSR, rare categories and calibration."),
    "insurance": dict(
        ref="mirichoi0218/insurance", file="insurance.csv", task="regression", target="charges",
        numeric=("age", "bmi", "children"), categorical=("sex", "smoker", "region"),
        purpose="Mixed numerical/categorical regression; regularization and linear-model limitations."),
}
DEFAULT_CACHE = Path(__file__).resolve().parents[1] / "data/kaggle"
MAX_BYTES = 64 * 1024 * 1024


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fetch_public(url):
    request = urllib.request.Request(url, headers={"User-Agent": "logitlab-benchmark/0.3"})
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            data = response.read(MAX_BYTES + 1)
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(
            "Kaggle download unavailable: check HTTPS access to www.kaggle.com and "
            "storage.googleapis.com in cloud environment settings. This public downloader "
            "does not need a GitHub token and never reads Kaggle credentials. " + str(error)) from error
    if len(data) > MAX_BYTES:
        raise ValueError("Download exceeds 64 MiB safety limit")
    return data


def unpack(payload, filename):
    if not zipfile.is_zipfile(io.BytesIO(payload)):
        raise ValueError("Expected a Kaggle ZIP archive; HTML/login responses are not data")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = archive.infolist()
        if sum(m.file_size for m in members) > MAX_BYTES:
            raise ValueError("Archive exceeds 64 MiB uncompressed limit")
        for member in members:
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in member.filename:
                raise ValueError("Unsafe archive path")
        matches = [m for m in members if m.filename == filename and not m.is_dir()]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {filename} in Kaggle archive")
        return archive.read(matches[0])


def download(case, cache=DEFAULT_CACHE, fetch=fetch_public):
    spec, directory = CATALOG[case], Path(cache) / case
    if directory.exists():
        _, _, manifest = load(case, cache)
        if manifest.get("evidence_kind") != "kaggle-download":
            raise ValueError("Refusing a protocol fixture as a downloaded Kaggle cache")
        return manifest
    metadata_url = "https://www.kaggle.com/api/v1/datasets/view/" + spec["ref"]
    metadata_bytes = fetch(metadata_url)
    try:
        metadata = json.loads(metadata_bytes)
        version, license_name = metadata["currentVersionNumber"], metadata["licenseName"]
        if isinstance(version, bool) or not isinstance(version, int) or version < 1 or not str(license_name or "").strip():
            raise ValueError("missing version/license")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Kaggle metadata must declare a positive version and licenseName; refusing to invent them") from error
    download_url = ("https://www.kaggle.com/api/v1/datasets/download/" + spec["ref"] +
                    f"?datasetVersionNumber={version}")
    payload = fetch(download_url)
    raw = unpack(payload, spec["file"])
    # Validate the actual schema/labels before publishing an atomic cache entry.
    parse(case, raw)
    manifest = dict(case=case, ref=spec["ref"], version=version, license=license_name,
                    file=spec["file"], source_url="https://www.kaggle.com/datasets/" + spec["ref"],
                    metadata_url=metadata_url, download_url=download_url,
                    downloaded_utc=datetime.now(timezone.utc).isoformat(),
                    archive_sha256=sha(payload), archive_bytes=len(payload),
                    csv_sha256=sha(raw), csv_bytes=len(raw), metadata_sha256=sha(metadata_bytes),
                    evidence_kind="kaggle-download")
    Path(cache).mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".download-", dir=cache))
    try:
        (temporary / spec["file"]).write_bytes(raw)
        (temporary / "metadata.json").write_bytes(metadata_bytes)
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.rename(directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def parse(case, raw):
    spec = CATALOG[case]
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Expected a UTF-8 CSV") from error
    reader = csv.DictReader(io.StringIO(text))
    expected = set(spec["numeric"] + spec["categorical"] + (spec["target"],))
    if not reader.fieldnames or not expected.issubset(reader.fieldnames):
        raise ValueError(f"CSV schema missing required columns: {sorted(expected - set(reader.fieldnames or []))}")
    rows, labels = [], []
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("Malformed CSV row")
        target = row[spec["target"]].strip().removesuffix(".")
        if spec["task"] == "classification":
            if target not in spec["labels"]:
                raise ValueError(f"Unknown target label: {target!r}")
            target = float(spec["labels"].index(target))
        else:
            try:
                target = float(target)
            except ValueError as error:
                raise ValueError("Non-numeric regression target") from error
            import math
            if not math.isfinite(target):
                raise ValueError("Non-finite regression target")
        rows.append({name: row[name].strip() for name in spec["numeric"] + spec["categorical"]})
        labels.append(target)
    if not rows:
        raise ValueError("Empty dataset")
    return rows, labels


def load(case, cache=DEFAULT_CACHE):
    import numpy as np
    spec, directory = CATALOG[case], Path(cache) / case
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
        raw = (directory / spec["file"]).read_bytes()
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Missing verified Kaggle cache for {case}; run python benchmarks/kaggle_data.py --cases {case}") from error
    if any(manifest.get(k) != value for k, value in (("case", case), ("ref", spec["ref"]), ("file", spec["file"]))):
        raise ValueError("Cache manifest identity mismatch")
    if sha(raw) != manifest.get("csv_sha256") or len(raw) != manifest.get("csv_bytes"):
        raise ValueError("Kaggle CSV checksum/length mismatch; do not reuse modified data")
    if not manifest.get("license") or not isinstance(manifest.get("version"), int) or manifest["version"] < 1:
        raise ValueError("Cache missing observed license/version")
    if manifest.get("evidence_kind") == "kaggle-download":
        metadata_raw = (directory / "metadata.json").read_bytes()
        if sha(metadata_raw) != manifest.get("metadata_sha256"):
            raise ValueError("Kaggle metadata checksum mismatch")
    rows, labels = parse(case, raw)
    return rows, np.asarray(labels, dtype=float), manifest


def prepare(case, rows, y, seed, scaling):
    import numpy as np
    from scipy import sparse
    from sklearn.preprocessing import OneHotEncoder
    from benchmarks import suite
    spec = CATALOG[case]
    if scaling not in ("raw", "rms"):
        raise ValueError("scaling must be raw or rms")
    y = np.asarray(y, dtype=float)
    if y.shape != (len(rows),) or not np.isfinite(y).all() or len(y) < 20:
        raise ValueError("Need at least 20 rows with finite one-dimensional targets")
    if spec["task"] == "classification":
        if set(y) != {0., 1.} or min(np.bincount(y.astype(int))) < 10:
            raise ValueError("Classification requires at least 10 rows in each binary class")
        splits = suite.split_indices(y, seed)
    else:
        indices = np.random.default_rng(seed + 100000).permutation(len(y))
        splits = tuple(np.split(indices, [int(.7 * len(y)), int(.85 * len(y))]))
    numeric = np.empty((len(rows), len(spec["numeric"])))
    for i, row in enumerate(rows):
        for j, name in enumerate(spec["numeric"]):
            value = str(row[name]).strip()
            try:
                numeric[i, j] = np.nan if value in ("", "?") else float(value)
            except ValueError as error:
                raise ValueError(f"Invalid numeric value in {name}") from error
            if value not in ("", "?") and not np.isfinite(numeric[i, j]):
                raise ValueError(f"Non-finite numeric value in {name}")
    train = numeric[splits[0]]
    medians = np.array([np.median(column[~np.isnan(column)]) if (~np.isnan(column)).any() else 0.
                        for column in train.T])
    numeric = np.where(np.isnan(numeric), medians, numeric)
    xs = [numeric[index] for index in splits]
    feature_names = list(spec["numeric"])
    if spec["categorical"]:
        categorical = np.array([[str(row[c]).strip() if str(row[c]).strip() not in ("", "?") else "__missing__"
                                 for c in spec["categorical"]] for row in rows], dtype=object)
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float64)
        encoder.fit(categorical[splits[0]])
        xs = [sparse.hstack((x, encoder.transform(categorical[index])), format="csr")
              for x, index in zip(xs, splits)]
        feature_names.extend(encoder.get_feature_names_out(spec["categorical"]).tolist())
    scale = suite.rms_scale(xs[0]) if scaling == "rms" else np.ones(xs[0].shape[1])
    xs = [suite.divide(x, scale) for x in xs]
    center, target_scale = 0., 1.
    if spec["task"] == "regression":
        center, target_scale = float(y[splits[0]].mean()), float(y[splits[0]].std())
        if target_scale == 0:
            target_scale = 1.
    ys = [(y[index] - center) / target_scale for index in splits]
    return dict(xs=xs, ys=ys, splits=splits, scale=scale, feature_names=feature_names,
                target_center=center, target_scale=target_scale,
                meta=dict(rows=len(y), cols=len(scale), train_rows=len(splits[0]),
                          validation_rows=len(splits[1]), test_rows=len(splits[2]),
                          storage="csr" if sparse.issparse(xs[0]) else "dense",
                          train_nnz=int(xs[0].nnz) if sparse.issparse(xs[0]) else int(xs[0].size),
                          split_sha256=suite.digest(*splits), scaler_sha256=suite.digest(scale, medians),
                          feature_names_sha256=sha(json.dumps(feature_names).encode()),
                          scale_ratio=float(scale.max() / scale.min()),
                          training_numeric_medians=medians.tolist(), target_center=center, target_scale=target_scale,
                          train_positive_rate=float(ys[0].mean()) if spec["task"] == "classification" else None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", choices=CATALOG, nargs="+", default=list(CATALOG))
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    for case in args.cases:
        try:
            manifest = download(case, args.cache)
        except (RuntimeError, ValueError, OSError, zipfile.BadZipFile) as error:
            parser.exit(1, f"{case}: {error}\n")
        print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
