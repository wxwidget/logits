"""Paired before/after benchmark with independent numerical acceptance checks.

Primary gate: geometric mean full-fit speedup >= 2 across ALL 18 predeclared
converged workloads. SGD/Adam use a separate fixed-budget track, never mixed
with convergence timing. Neither accuracy tolerances nor cases are auto-tuned.
"""

import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from scipy import sparse
from scipy.special import expit

PRIMARY = (
    "logistic:gd",
    "logistic:lbfgs",
    "logistic:newton-cg",
    "linear:cg",
    "linear:proximal-gd",
    "linear:fista",
    "glm:gaussian",
    "glm:binomial",
    "glm:poisson",
)
SECONDARY = ("logistic:sgd", "logistic:adam")


def dataset(storage, seed, quick=False):
    rng = np.random.default_rng(seed)
    n, p = (240, 8) if quick else (24000, 96 if storage == "dense" else 2000)
    if storage == "dense":
        X = rng.normal(size=(n, p))
    else:
        density = 0.2 if quick else 0.01
        X = sparse.random(
            n,
            p,
            density=density,
            format="csr",
            random_state=rng,
            data_rvs=lambda size: rng.normal(size=size) / np.sqrt(density),
        )
        X.sort_indices()
    beta = rng.normal(size=p) / np.sqrt(p)
    signal = np.asarray(X @ beta).ravel() * 0.6 + 0.2
    targets = dict(
        gaussian=signal + rng.normal(scale=0.5, size=n),
        binomial=rng.binomial(1, expit(signal)).astype(float),
        poisson=rng.poisson(np.exp(signal)).astype(float),
    )
    return X, targets


def independent(X, y, theta, family, l1, l2):
    eta = np.asarray(X @ theta[:-1]).ravel() + theta[-1]
    if family == "gaussian":
        loss, residual = 0.5 * (eta - y) ** 2, eta - y
    elif family == "binomial":
        loss, residual = np.logaddexp(0, eta) - y * eta, expit(eta) - y
    else:
        loss, residual = np.exp(eta) - y * eta, np.exp(eta) - y
    objective = float(
        loss.mean()
        + l1 * np.abs(theta[:-1]).sum()
        + 0.5 * l2 * (theta[:-1] @ theta[:-1])
    )
    gradient = np.r_[
        np.asarray(X.T @ residual).ravel() / len(y) + l2 * theta[:-1], residual.mean()
    ]
    proposal = theta - gradient
    proposal[:-1] = np.sign(proposal[:-1]) * np.maximum(0, np.abs(proposal[:-1]) - l1)
    return objective, float(np.linalg.norm(theta - proposal))


def worker(job):
    # Import exactly the requested Python/native artifact; no installed fallback.
    sys.path.insert(0, job["package"])
    import logitlab
    from logitlab import (
        _core,
        LinearRegression,
        LogisticRegression,
        GeneralizedLinearModel,
    )

    if (
        not Path(logitlab.__file__)
        .resolve()
        .is_relative_to(Path(job["package"]).resolve())
    ):
        raise RuntimeError("Wrong package selected")
    kind, solver = job["case"].split(":")
    family = (
        solver if kind == "glm" else "binomial" if kind == "logistic" else "gaussian"
    )
    l1 = 0.01 if kind == "linear" and solver != "cg" else 0.0
    l2 = 0.1
    X, targets = dataset(job["storage"], job["seed"], job["quick"])
    y = targets[family]
    cut = int(0.8 * len(y))
    train, test = X[:cut], X[cut:]
    yt, ye = y[:cut], y[cut:]
    arrays = (X.data, X.indices, X.indptr, y) if sparse.issparse(X) else (X, y)
    digest = hashlib.sha256()
    for a in arrays:
        digest.update(np.ascontiguousarray(a).tobytes())

    def model():
        if kind == "glm":
            return GeneralizedLinearModel(
                family=family, l2=l2, max_iter=300, tolerance=1e-6
            )
        if kind == "linear":
            return LinearRegression(
                solver=solver, l1=l1, l2=l2, max_iter=1000, tolerance=1e-6
            )
        return LogisticRegression(
            solver=solver,
            l2=l2,
            max_epochs=30 if job["secondary"] else 300,
            learning_rate=0.003
            if solver == "adam"
            else 0.03
            if solver == "sgd"
            else 1.0,
            tolerance=1e-6,
            seed=job["seed"],
        )

    model().fit(train[:64], yt[:64])
    rows = []
    for repeat in range(job["repeats"]):
        estimator = model()
        start = time.perf_counter()
        estimator.fit(train, yt)
        seconds = time.perf_counter() - start
        theta = np.r_[estimator.coef_, estimator.intercept_]
        objective, stationarity = independent(train, yt, theta, family, l1, l2)
        predict = (
            (lambda: estimator.predict_proba(test)[:, 1])
            if kind == "logistic"
            else (lambda: estimator.predict(test))
        )
        predict()
        durations = []
        for _ in range(7):
            start = time.perf_counter()
            prediction = predict()
            durations.append(time.perf_counter() - start)
        if family == "binomial":
            p = np.clip(prediction, 1e-15, 1 - 1e-15)
            quality = float(-np.mean(ye * np.log(p) + (1 - ye) * np.log1p(-p)))
        else:
            quality = float(np.mean((prediction - ye) ** 2))
        history = estimator.history_
        row = dict(
            case=job["case"],
            storage=job["storage"],
            seed=job["seed"],
            repeat=repeat,
            secondary=job["secondary"],
            fit_seconds=seconds,
            cpp_seconds=history[-1]["seconds"],
            prediction_seconds=float(np.median(durations)),
            status=estimator.status_,
            converged=estimator.converged_,
            objective=objective,
            stationarity=stationarity,
            heldout_quality=quality,
            rows=len(y),
            train_rows=cut,
            cols=X.shape[1],
            nnz=int(train.nnz) if sparse.issparse(train) else train.size,
            data_sha256=digest.hexdigest(),
            work=estimator.work_,
        )
        if repeat == 0:
            row.update(parameters=theta.tolist(), history=history)
        rows.append(row)
    peak = (
        int(
            next(
                line.split()[1]
                for line in Path("/proc/self/status").read_text().splitlines()
                if line.startswith("VmHWM:")
            )
        )
        / 1024
    )
    for row in rows:
        row["worker_peak_rss_mib"] = peak
    return dict(
        rows=rows,
        artifact=dict(
            module=str(_core.__file__),
            module_sha256=hashlib.sha256(Path(_core.__file__).read_bytes()).hexdigest(),
            python=str(Path(logitlab.__file__).resolve()),
            runtime=getattr(
                _core, "_runtime_features", lambda: {"simd": "compiler_default"}
            )(),
        ),
    )


def compare(before, after):
    keys = ("case", "storage", "seed", "repeat")

    def index(rows):
        result = {tuple(r[k] for k in keys): r for r in rows}
        if len(result) != len(rows):
            raise ValueError("duplicate measurement")
        return result

    a, b = index(before), index(after)
    if set(a) != set(b) or not a:
        raise ValueError("Incomplete paired results")
    expected = {
        (case, storage, seed, repeat)
        for case in PRIMARY + SECONDARY
        for storage in ("dense", "csr")
        for seed in {r["seed"] for r in before}
        for repeat in range(1 + max(r["repeat"] for r in before))
    }
    if set(a) != expected:
        raise ValueError("Incomplete declared workload matrix")
    for key, old in a.items():
        new = b[key]
        for row in (old, new):
            if not all(
                np.isfinite(row[name])
                for name in (
                    "objective",
                    "stationarity",
                    "heldout_quality",
                    "fit_seconds",
                )
            ):
                raise ValueError("Non-finite measurement")
            if row["fit_seconds"] <= 0:
                raise ValueError("Non-positive timing")
        if old["data_sha256"] != new["data_sha256"]:
            raise ValueError("Different datasets")
        if not old["secondary"] and not all(
            r["converged"] and r["stationarity"] <= 1.00001e-6 for r in (old, new)
        ):
            raise ValueError(f"Unmatched convergence target: {key}")
        if abs(new["objective"] - old["objective"]) > 1e-9:
            raise ValueError(f"Different objectives: {key}")
        if abs(new["heldout_quality"] - old["heldout_quality"]) > 2e-5:
            raise ValueError(f"Different predictive quality: {key}")
        if "parameters" in old and not np.allclose(
            old["parameters"], new["parameters"], atol=3e-5, rtol=1e-5
        ):
            raise ValueError(f"Different parameters: {key}")
    summaries = []
    for case, storage in sorted({(r["case"], r["storage"]) for r in before}):
        entry = dict(case=case, storage=storage, secondary=case in SECONDARY)
        for label, rows in (("before", before), ("after", after)):
            selected = [
                r for r in rows if r["case"] == case and r["storage"] == storage
            ]
            medians = [
                np.median([r["fit_seconds"] for r in selected if r["seed"] == seed])
                for seed in sorted({r["seed"] for r in selected})
            ]
            entry[label] = dict(
                zip(
                    ("q1", "median", "q3"),
                    map(float, np.quantile(medians, [0.25, 0.5, 0.75])),
                )
            )
        entry["speedup"] = entry["before"]["median"] / entry["after"]["median"]
        summaries.append(entry)
    primary = [s["speedup"] for s in summaries if not s["secondary"]]
    geometric = float(np.exp(np.mean(np.log(primary))))
    return dict(
        primary_geometric_speedup=geometric,
        target=2.0,
        passed=geometric >= 2.0,
        cases=summaries,
    )


def source_hashes(root):
    patterns = (
        "src/**/*.cpp",
        "src/**/*.hpp",
        "include/**/*.hpp",
        "bindings/*.cpp",
        "bindings/*.hpp",
        "python/logitlab/*.py",
        "CMakeLists.txt",
        "pyproject.toml",
        "benchmarks/refactor.py",
    )
    files = sorted(
        {p for pattern in patterns for p in root.glob(pattern) if p.is_file()}
    )
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }


def build_metadata(directory, source):
    if directory is None:
        return None
    cache = {}
    for line in (directory / "CMakeCache.txt").read_text().splitlines():
        if line.startswith(("//", "#")) or "=" not in line or ":" not in line:
            continue
        key, value = line.split("=", 1)
        cache[key.split(":")[0]] = value
    if (
        source is not None
        and Path(cache["CMAKE_HOME_DIRECTORY"]).resolve() != source.resolve()
    ):
        raise ValueError("Build cache and declared source do not match")
    names = (
        "CMAKE_CXX_COMPILER",
        "CMAKE_BUILD_TYPE",
        "CMAKE_CXX_FLAGS",
        "CMAKE_CXX_FLAGS_RELEASE",
        "LOGITLAB_NATIVE",
        "LOGITLAB_SIMD",
        "LOGITLAB_VECTOR_MATH",
        "LOGITLAB_MVEC_LIBRARY",
    )
    result = {name: cache.get(name) for name in names}
    lib = cache.get("LOGITLAB_MVEC_LIBRARY")
    if lib and Path(lib).is_file():
        result["libmvec_sha256"] = hashlib.sha256(Path(lib).read_bytes()).hexdigest()
    return result


def machine_metadata(cpu):
    model = next(
        line.split(":", 1)[1].strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines()
        if line.startswith("model name")
    )
    return dict(
        cpu=model,
        affinity=sorted(os.sched_getaffinity(0)),
        selected_cpu=cpu,
        platform=platform.platform(),
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=__import__("scipy").__version__,
        libc=platform.libc_ver(),
        measured_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        threads={
            name: os.environ[name]
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        compiler=subprocess.run(
            ["g++", "--version"], capture_output=True, text=True, check=True
        ).stdout,
    )


def run(args):
    cpu = min(os.sched_getaffinity(0)) if args.cpu is None else args.cpu
    os.sched_setaffinity(0, {cpu})
    args.output.mkdir(parents=True, exist_ok=True)
    data = dict(
        protocol="18 fixed converged cases; 4 separate stochastic fixed-budget cases",
        config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        before=[],
        after=[],
        artifacts={},
        machine=machine_metadata(cpu),
        sources={},
        builds={},
    )
    for label in ("before", "after"):
        source = getattr(args, label + "_source")
        data["sources"][label] = source_hashes(source) if source else None
        data["builds"][label] = build_metadata(getattr(args, label + "_build"), source)
    for i, (storage, case, seed) in enumerate(
        (s, c, seed)
        for s in ("dense", "csr")
        for c in PRIMARY + SECONDARY
        for seed in args.seeds
    ):
        for label in ("before", "after") if i % 2 == 0 else ("after", "before"):
            job = dict(
                package=str(getattr(args, label).resolve()),
                case=case,
                storage=storage,
                seed=seed,
                quick=args.quick,
                repeats=args.repeats,
                secondary=case in SECONDARY,
            )
            p = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--worker"],
                input=json.dumps(job),
                capture_output=True,
                text=True,
            )
            if p.returncode:
                raise RuntimeError(f"{label}/{case}/{storage}: {p.stderr}")
            result = json.loads(p.stdout)
            data[label].extend(result["rows"])
            previous = data["artifacts"].get(label)
            if previous and previous != result["artifact"]:
                raise ValueError("Artifact changed during measurement")
            data["artifacts"][label] = result["artifact"]
            (args.output / "results.json").write_text(json.dumps(data, indent=2) + "\n")
        print(storage, case, seed, flush=True)
    data["comparison"] = compare(data["before"], data["after"])
    for label in ("before", "after"):
        source = getattr(args, label + "_source")
        if source and source_hashes(source) != data["sources"][label]:
            raise ValueError("Source changed during measurement")
    data["eligible_for_performance_claim"] = (
        not args.quick
        and len(args.seeds) >= 3
        and args.repeats >= 3
        and all(data["builds"].values())
        and all(data["sources"].values())
        and data["artifacts"]["before"]["module_sha256"]
        != data["artifacts"]["after"]["module_sha256"]
    )
    (args.output / "results.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data["comparison"], indent=2))


def main():
    if sys.argv[1:] == ["--worker"]:
        print(json.dumps(worker(json.load(sys.stdin)), allow_nan=False))
        return
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--before",
        type=Path,
        required=True,
        help="Directory containing baseline logitlab package",
    )
    p.add_argument(
        "--after",
        type=Path,
        required=True,
        help="Directory containing candidate logitlab package",
    )
    p.add_argument("--output", type=Path, default=Path("build/refactor-benchmark"))
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--quick", action="store_true")
    p.add_argument(
        "--cpu",
        type=int,
        help="Pin both artifacts to this CPU; default is first allowed CPU",
    )
    for label in ("before", "after"):
        p.add_argument("--" + label + "-source", type=Path)
        p.add_argument("--" + label + "-build", type=Path)
    args = p.parse_args()
    if args.repeats < 1 or len(args.seeds) != len(set(args.seeds)):
        p.error("positive repeats and unique seeds required")
    run(args)


if __name__ == "__main__":
    main()
