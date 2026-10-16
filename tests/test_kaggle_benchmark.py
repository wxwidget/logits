"""Offline protocol fixtures are not downloaded Kaggle measurements."""
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks import kaggle_data, kaggle_suite, kaggle_report


def archive(name, content):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as z:
        z.writestr(name, content)
    return stream.getvalue()


def fixture_cache(root, case="insurance", rows=100):
    spec = kaggle_data.CATALOG[case]
    directory = root / case
    directory.mkdir(parents=True)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(spec["numeric"] + spec["categorical"]) + [spec["target"]])
    writer.writeheader()
    rng = np.random.default_rng(12)
    for i in range(rows):
        row = {name: float(rng.uniform(1, 8)) for name in spec["numeric"]}
        row.update({name: ["a", "b", "c"][i % 3] for name in spec["categorical"]})
        row[spec["target"]] = (100 + 2 * row[spec["numeric"][0]] + rng.normal()
                               if spec["task"] == "regression" else spec["labels"][i % 2])
        writer.writerow(row)
    raw = stream.getvalue().encode()
    (directory / spec["file"]).write_bytes(raw)
    manifest = dict(case=case, ref=spec["ref"], version=1, license="TEST FIXTURE ONLY",
                    source_url="https://www.kaggle.com/datasets/" + spec["ref"],
                    csv_sha256=hashlib.sha256(raw).hexdigest(), csv_bytes=len(raw),
                    file=spec["file"], evidence_kind="protocol-fixture")
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


def test_download_pins_version_and_records_observed_license(tmp_path):
    calls = []
    payload = b"age,bmi,children,sex,smoker,region,charges\n1,2,0,female,no,north,3\n"
    def fetch(url):
        calls.append(url)
        if "/view/" in url:
            return json.dumps(dict(currentVersionNumber=7, licenseName="CC0: Public Domain")).encode()
        return archive("insurance.csv", payload)
    manifest = kaggle_data.download("insurance", tmp_path, fetch=fetch)
    assert "datasetVersionNumber=7" in calls[1]
    assert manifest["license"] == "CC0: Public Domain"
    assert manifest["csv_sha256"] == hashlib.sha256(payload).hexdigest()
    assert manifest["evidence_kind"] == "kaggle-download"
    kaggle_data.load("insurance", tmp_path)
    # A valid cache does not perform network calls; corrupted cache fails closed.
    kaggle_data.download("insurance", tmp_path, fetch=lambda _: pytest.fail("unexpected network"))
    (tmp_path / "insurance" / "insurance.csv").write_text("corrupt")
    with pytest.raises(ValueError, match="checksum"):
        kaggle_data.load("insurance", tmp_path)


@pytest.mark.parametrize("payload", [b"<html>login required</html>", archive("../insurance.csv", b"bad"),
                                     archive("different.csv", b"bad")])
def test_download_rejects_html_and_unsafe_or_wrong_archive(tmp_path, payload):
    def fetch(url):
        return (b'{"currentVersionNumber":1,"licenseName":"CC0"}'
                if "/view/" in url else payload)
    with pytest.raises(ValueError):
        kaggle_data.download("insurance", tmp_path, fetch=fetch)
    assert not (tmp_path / "insurance" / "manifest.json").exists()


def test_missing_license_or_version_is_not_invented(tmp_path):
    for metadata in ({"currentVersionNumber": 1}, {"licenseName": "CC0"}):
        with pytest.raises(ValueError, match="metadata"):
            kaggle_data.download("insurance", tmp_path, fetch=lambda _: json.dumps(metadata).encode())


@pytest.mark.parametrize("case", list(kaggle_data.CATALOG))
def test_prepare_disjoint_training_only_and_csr(tmp_path, case):
    pytest.importorskip("sklearn")
    fixture_cache(tmp_path, case)
    rows, y, manifest = kaggle_data.load(case, tmp_path)
    raw = kaggle_data.prepare(case, rows, y, seed=42, scaling="raw")
    scaled = kaggle_data.prepare(case, rows, y, seed=42, scaling="rms")
    splits = raw["splits"]
    assert set.intersection(*(set(s) for s in splits)) == set()
    assert sum(map(len, splits)) == len(set(np.concatenate(splits))) == len(y)
    assert raw["meta"]["split_sha256"] == scaled["meta"]["split_sha256"]
    for x, z in zip(raw["xs"], scaled["xs"]):
        if case != "breast_cancer":
            assert sparse.isspmatrix_csr(x) and sparse.isspmatrix_csr(z)
            x, z = x.toarray(), z.toarray()
        np.testing.assert_allclose(z, x / scaled["scale"])
    # Change only held-out covariates/labels. No fitted training transform changes.
    altered = [dict(r) for r in rows]
    for index in np.r_[splits[1], splits[2]]:
        altered[index][kaggle_data.CATALOG[case]["numeric"][0]] = "1000000000"
        for col in kaggle_data.CATALOG[case]["categorical"]:
            altered[index][col] = "heldout-only-category"
    changed_y = y.copy()
    if case == "insurance":
        changed_y[splits[2]] *= 1e8
    again = kaggle_data.prepare(case, altered, changed_y, seed=42, scaling="rms")
    np.testing.assert_array_equal(scaled["scale"], again["scale"])
    np.testing.assert_array_equal(scaled["ys"][0], again["ys"][0])
    assert scaled["feature_names"] == again["feature_names"]
    assert not any("heldout-only" in name for name in again["feature_names"])


def test_schema_target_and_checksum_guards(tmp_path):
    fixture_cache(tmp_path, "adult")
    rows, y, _ = kaggle_data.load("adult", tmp_path)
    assert set(y) == {0., 1.}
    assert all("income" not in row and "fnlwgt" not in row for row in rows)
    rows[0]["age"] = "nan"
    with pytest.raises(ValueError, match="finite|numeric"):
        kaggle_data.prepare("adult", rows, y, 42, "rms")


def test_average_precision_and_constant_baseline():
    pytest.importorskip("sklearn")
    from sklearn.metrics import average_precision_score
    y, p = np.array([0., 0., 1., 0., 1.]), np.array([.1, .2, .2, .4, .9])
    assert kaggle_suite.classification_metrics(y, p)["average_precision"] == pytest.approx(average_precision_score(y, p))
    assert kaggle_suite.classification_metrics(y, np.full(5, .4))["average_precision"] == pytest.approx(.4)


def test_fixture_runner_report_and_incomplete_rejection(tmp_path):
    pytest.importorskip("sklearn")
    cache, output = tmp_path / "cache", tmp_path / "output"
    fixture_cache(cache, "insurance", rows=80)
    fixture_cache(cache, "breast_cancer", rows=80)
    fixture_cache(cache, "adult", rows=80)
    command = [sys.executable, str(Path(kaggle_suite.__file__)), "--cases", "insurance", "breast_cancer", "adult",
               "--cache", str(cache), "--output", str(output), "--seeds", "42", "--repeats", "1",
               "--iterations", "2", "--allow-fixtures"]
    subprocess.run(command, check=True, capture_output=True, text=True)
    data = json.loads((output / "results.json").read_text())
    assert data["evidence_kind"] == "protocol-fixture"
    kaggle_report.validate_complete(data)
    assert {r["solver"] for r in data["results"]} == {"cg", "proximal-gd", "fista", "gd", "sgd", "adam", "lbfgs", "newton-cg"}
    assert all(r["fit_seconds"] > 0 for r in data["results"])
    assert all(r["test"] for r in data["results"])
    assert (output / "metrics.csv").is_file() and (output / "traces.csv").is_file()
    for chinese in (False, True):
        report = kaggle_report.render(data, chinese)
        assert "PROTOCOL FIXTURE" in report and "average_precision" in report and "insurance" in report
    subprocess.run([sys.executable, str(Path(kaggle_report.__file__)), "--input", str(output),
                    "--output", str(tmp_path / "report")], check=True, capture_output=True, text=True)
    for name in ("kaggle.en.md", "kaggle.zh-CN.md", "quality-cost.png", "summary.json", "report-manifest.json"):
        assert (tmp_path / "report" / name).stat().st_size > 50
    with pytest.raises(ValueError, match="Incomplete"):
        kaggle_report.validate_complete(data | {"results": data["results"][:-1]})
    data["results"].append(data["results"][0])
    with pytest.raises(ValueError, match="Incomplete"):
        kaggle_report.validate_complete(data)


def test_refuse_fixture_as_real_evidence(tmp_path):
    fixture_cache(tmp_path, "insurance")
    with pytest.raises(ValueError, match="fixture"):
        kaggle_suite.load_prepared("insurance", tmp_path, 42, "rms")
    with pytest.raises(ValueError, match="fixture"):
        kaggle_data.download("insurance", tmp_path, fetch=lambda _: pytest.fail("unexpected network"))


@pytest.mark.parametrize("case", ["insurance", "breast_cancer"])
def test_test_targets_cannot_select_regularization(tmp_path, case):
    pytest.importorskip("sklearn")
    fixture_cache(tmp_path, case)
    prepared = kaggle_suite.load_prepared(case, tmp_path, 42, "rms", allow_fixtures=True)
    task = kaggle_data.CATALOG[case]["task"]
    model = "ridge" if task == "regression" else "logistic"
    xs, ys = prepared["xs"], prepared["ys"]
    a, trials_a = kaggle_suite.select_model(task, model, xs, ys, prepared["scale"])
    changed = [part.copy() for part in ys]
    changed[2][:] = 1e30
    b, trials_b = kaggle_suite.select_model(task, model, xs, changed, prepared["scale"])
    assert a[0] == b[0]
    assert [t["validation"] for t in trials_a] == [t["validation"] for t in trials_b]


def test_missing_values_use_training_medians(tmp_path):
    pytest.importorskip("sklearn")
    fixture_cache(tmp_path)
    rows, y, _ = kaggle_data.load("insurance", tmp_path)
    splits = kaggle_data.prepare("insurance", rows, y, 42, "raw")["splits"]
    for index in splits[1]:
        rows[index]["age"] = "?"
    rows[splits[0][0]]["age"] = ""
    prepared = kaggle_data.prepare("insurance", rows, y, 42, "raw")
    median = np.median([float(rows[i]["age"]) for i in splits[0][1:]])
    np.testing.assert_allclose(prepared["xs"][1].toarray()[:, 0], median)
    assert prepared["xs"][0][0, 0] == pytest.approx(median)


def test_aggregate_repeats_within_seed_before_across_seeds():
    base = dict(case="insurance", scaling="rms", model="ols", solver="cg", status="converged", test={})
    rows = [base | dict(seed=42, fit_seconds=1.) for _ in range(3)]
    rows += [base | dict(seed=43, fit_seconds=100.)]
    summary, = kaggle_report.summarize({"results": rows})
    assert summary["fit_seconds"]["median"] == 50.5


def test_metadata_checksum_and_target_schema_validation(tmp_path):
    payload = b"age,bmi,children,sex,smoker,region,charges\n1,2,0,female,no,north,3\n"
    def fetch(url):
        return (b'{"currentVersionNumber":1,"licenseName":"CC0"}'
                if "/view/" in url else archive("insurance.csv", payload))
    kaggle_data.download("insurance", tmp_path, fetch=fetch)
    (tmp_path / "insurance" / "metadata.json").write_text("tampered")
    with pytest.raises(ValueError, match="metadata checksum"):
        kaggle_data.load("insurance", tmp_path)
    with pytest.raises(ValueError, match="schema"):
        kaggle_data.parse("insurance", b"age,charges\n1,3\n")
    with pytest.raises(ValueError, match="target"):
        kaggle_data.parse("insurance", payload.replace(b",3\n", b",nan\n"))
