"""Controlled optimizer study. Run from the checkout; outputs contain all trials.

The five solvers fit one model family. RMS scaling never centers sparse data.
Calibration minimizes TRAIN objective; test labels are never used for selection.
"""
import os

# Set before NumPy/SciPy import, including in child workers.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import contextlib
import csv
import hashlib
import io
from importlib.metadata import version
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

from logitlab import LogisticRegression
from logitlab import _core

ROOT = Path(__file__).resolve().parents[1]
SOLVERS = ("gd", "sgd", "adam", "lbfgs", "newton-cg")
CASES = ("dense_control", "dense_multiscale", "dense_correlated", "sparse_wide", "heart")
RATES = {"gd": (.05, .5, 5.), "sgd": (.001, .01, .1),
         "adam": (.0001, .001, .01), "lbfgs": (.05,), "newton-cg": (.05,)}


def digest(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def dataset(case, seed, quick=False):
    rng = np.random.default_rng(seed)
    if case == "heart":
        # Existing LIBSVM fixture; missing coordinates are zero, +1 -> 1.
        lines = (ROOT / "data/heart_scale").read_text().splitlines()
        X, y = np.zeros((len(lines), 13)), np.zeros(len(lines))
        for i, line in enumerate(lines):
            tokens = line.split()
            y[i] = float(tokens[0]) == 1
            for token in tokens[1:]:
                j, value = token.split(":")
                X[i, int(j) - 1] = float(value)
        return X, y
    n, p = (160, 12) if quick else (6000, 48)
    if case == "sparse_wide":
        n, p = (220, 24) if quick else (12000, 1500)
        X = sparse.random(n, p, density=.08 if quick else .01, format="csr",
                          random_state=rng, data_rvs=lambda k: rng.normal(size=k))
        X.sort_indices()
        signal = np.asarray(X @ rng.normal(size=p)).ravel() / np.sqrt(p * (.08 if quick else .01))
        factors = np.geomspace(.01, 100, p)
        X = X.multiply(factors).tocsr()
    else:
        X = rng.normal(size=(n, p))
        if case == "dense_correlated":
            latent = rng.normal(size=(n, max(1, p // 4)))
            X = np.repeat(latent, 4, axis=1)[:, :p] + .05 * X
            # Opposing correlated coefficients expose poorly resolved directions.
            truth = np.tile([1., -1., .5, -.5], p // 4) / .05
            signal = X @ truth / np.sqrt(p)
        else:
            signal = X @ rng.normal(size=p) / np.sqrt(p)
        if case == "dense_multiscale":
            X *= np.geomspace(.01, 100, p)
    y = (rng.random(n) < expit(2 * signal - .3)).astype(float)
    return X, y


def split_indices(y, seed):
    rng = np.random.default_rng(seed + 100000)
    parts = [[], [], []]
    for label in (0, 1):
        indices = rng.permutation(np.flatnonzero(y == label))
        a, b = int(.7 * len(indices)), int(.85 * len(indices))
        for out, chunk in zip(parts, (indices[:a], indices[a:b], indices[b:])):
            out.extend(chunk)
    return tuple(rng.permutation(np.asarray(part, dtype=int)) for part in parts)


def rms_scale(train):
    squares = train.power(2).mean(axis=0) if sparse.issparse(train) else np.mean(train**2, axis=0)
    scale = np.sqrt(np.asarray(squares).ravel())
    scale[scale == 0] = 1
    return scale


def divide(X, scale):
    return X.multiply(1 / scale).tocsr() if sparse.issparse(X) else X / scale


def prepare(case, seed, scaling, quick):
    X, y = dataset(case, seed, quick)
    splits = split_indices(y, seed)
    xs = [X[index] for index in splits]
    ys = [y[index] for index in splits]
    scale = rms_scale(xs[0])
    arrays = (X.data, X.indices, X.indptr, y) if sparse.issparse(X) else (X, y)
    meta = dict(rows=X.shape[0], cols=X.shape[1], train_rows=len(ys[0]),
                validation_rows=len(ys[1]), test_rows=len(ys[2]),
                train_nnz=int(xs[0].nnz) if sparse.issparse(X) else int(xs[0].size),
                storage="csr" if sparse.issparse(X) else "dense",
                data_sha256=digest(*arrays), split_sha256=digest(*splits),
                scale_sha256=digest(scale), scale_ratio=float(scale.max() / scale.min()))
    if scaling == "rms":
        xs = [divide(x, scale) for x in xs]
    return xs, ys, meta


def objective(X, y, theta, penalty):
    z = np.asarray(X @ theta[:-1]).ravel() + theta[-1]
    residual = expit(z) - y
    loss = np.mean(np.logaddexp(0, (1 - 2 * y) * z)) + .5 * np.sum(penalty * theta[:-1]**2)
    gradient = np.r_[np.asarray(X.T @ residual).ravel() / len(y) + penalty * theta[:-1],
                     np.mean(residual)]
    return float(loss), gradient


def reference(X, y, l2):
    # Coordinate preconditioning only: the penalty transforms with the weights.
    # For raw X this remains exactly l2/2 ||w_raw||^2, NOT an RMS-model penalty.
    scale = rms_scale(X)
    Z = divide(X, scale)
    penalty = l2 / scale**2
    start = time.perf_counter()
    fun = lambda theta: objective(Z, y, theta, penalty)

    def hvp(theta, vector):
        q = expit(np.asarray(Z @ theta[:-1]).ravel() + theta[-1])
        v = np.asarray(Z @ vector[:-1]).ravel() + vector[-1]
        r = q * (1 - q) * v / len(y)
        return np.r_[np.asarray(Z.T @ r).ravel() + penalty * vector[:-1], np.sum(r)]

    initial = np.zeros(X.shape[1] + 1)
    first = minimize(fun, initial, jac=True, hessp=hvp, method="Newton-CG",
                     options={"xtol": 1e-10, "maxiter": 500})
    result = first
    if np.linalg.norm(fun(first.x)[1]) > 1e-7:
        second = minimize(fun, first.x, jac=True, method="L-BFGS-B",
                          options={"ftol": 1e-14, "gtol": 1e-9, "maxiter": 5000, "maxls": 40})
        if second.fun < first.fun:
            result = second
    loss, grad = fun(result.x)
    return dict(objective=loss, gradient_norm_preconditioned=float(np.linalg.norm(grad)),
                verified=bool(np.linalg.norm(grad) <= 1e-5), scipy_success=bool(result.success),
                message=str(result.message), seconds=time.perf_counter() - start,
                nfev=int(first.nfev + (second.nfev if 'second' in locals() else 0)))


def metrics(y, probability):
    p = np.clip(probability, 1e-15, 1 - 1e-15)
    positive = y == 1
    n1, n0 = int(positive.sum()), int((~positive).sum())
    auc = ((rankdata(probability)[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
           if n1 and n0 else None)
    return dict(log_loss=float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p))),
                accuracy=float(np.mean((probability >= .5) == y)),
                roc_auc=float(auc) if auc is not None else None,
                brier=float(np.mean((probability - y)**2)))


def fit_once(X, y, options):
    start = time.perf_counter()
    try:
        model = LogisticRegression(**options).fit(X, y)
        return model, time.perf_counter() - start, None
    except RuntimeError as error:
        return None, time.perf_counter() - start, str(error)


def calibrate(X, y, options):
    trials = []
    for rate in RATES[options["solver"]]:
        candidate = options | {"learning_rate": rate}
        model, seconds, error = fit_once(X, y, candidate)
        trials.append(dict(learning_rate=rate, fit_seconds=seconds, error=error,
                           objective=model.history_[-1]["loss"] if model else None,
                           status=model.status_ if model else "error"))
    good = [trial for trial in trials if trial["objective"] is not None]
    best = min(good, key=lambda trial: trial["objective"]) if good else trials[0]
    return options | {"learning_rate": best["learning_rate"]}, trials


def first_target(history, ref, target_gap):
    if not ref["verified"]:
        return None
    # Materially better than the reference invalidates this comparison.
    if any(h["loss"] < ref["objective"] - 1e-7 for h in history):
        return None
    for h in history:
        if h["loss"] - ref["objective"] <= target_gap:
            return {key: h[key] for key in ("epoch", "seconds", "row_visits")}
    return None


def worker(job):
    xs, ys, meta = prepare(job["case"], job["seed"], job["scaling"], job["quick"])
    options = job["options"]
    # Small API warm-up, excluded from measured fits. Full calibration is separate.
    fit_once(xs[0][:32], ys[0][:32], options | {"max_epochs": 1})
    rows, traces = [], []
    for repeat in range(job["repeats"]):
        model, elapsed, error = fit_once(xs[0], ys[0], options)
        row = dict(repeat=repeat, fit_seconds=elapsed, error=error, **meta)
        if model is None:
            rows.append(row | dict(status="error", target=None))
            continue
        # Warm prediction separately, then repeat the Python-facing prediction API.
        model.predict_proba(xs[2])
        durations = []
        for _ in range(7):
            start = time.perf_counter()
            probabilities = model.predict_proba(xs[2])[:, 1]
            durations.append(time.perf_counter() - start)
        prediction_seconds = float(np.median(durations))
        final = model.history_[-1]
        gap = final["loss"] - job["reference"]["objective"]
        valid = job["reference"]["verified"] and gap >= -1e-7
        row.update(status=model.status_, epochs=final["epoch"], train_objective=final["loss"],
                   gradient_norm=final["gradient_norm"], cpp_seconds=final["seconds"],
                   objective_gap=max(0., gap) if valid else None,
                   target=first_target(model.history_, job["reference"], job["target_gap"]),
                   reference_valid=valid, **model.work_,
                   prediction_seconds=prediction_seconds,
                   prediction_rows_per_second=len(ys[2]) / prediction_seconds,
                   validation=metrics(ys[1], model.predict_proba(xs[1])[:, 1]),
                   test=metrics(ys[2], probabilities))
        rows.append(row)
        traces.extend(dict(repeat=repeat, **h) for h in model.history_)
        del model
    # Linux ru_maxrss can inherit the parent's pre-exec high-water mark. VmHWM
    # belongs to this executable's address space, so it avoids that contamination.
    status = Path("/proc/self/status")
    if status.exists():
        peak = int(next(line.split()[1] for line in status.read_text().splitlines()
                        if line.startswith("VmHWM:"))) / 1024
        rss_method = "Linux /proc/self/status VmHWM (post-exec address space)"
    else:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
            1024**2 if sys.platform == "darwin" else 1024)
        rss_method = "getrusage whole-process high-water mark; may include launcher"
    for row in rows:
        row["worker_peak_rss_mib"] = peak
        row["rss_method"] = rss_method
    return dict(rows=rows, traces=traces)


def machine_metadata(build_cache=None):
    with io.StringIO() as stream, contextlib.redirect_stdout(stream):
        np.show_config()
        blas = stream.getvalue()
    def command(args):
        return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()
    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), platform.processor())
    build = None
    if build_cache is not None:
        cache = (build_cache / "CMakeCache.txt").read_text().splitlines()
        names = ("CMAKE_BUILD_TYPE", "CMAKE_CXX_COMPILER", "CMAKE_CXX_FLAGS",
                 "CMAKE_CXX_FLAGS_RELEASE", "LOGITLAB_NATIVE", "FETCHCONTENT_SOURCE_DIR_ABSEIL",
                 "FETCHCONTENT_SOURCE_DIR_PYBIND11")
        build = {line.split(":", 1)[0]: line.split("=", 1)[1] for line in cache
                 if not line.startswith(("#", "//")) and "=" in line
                 and line.split(":", 1)[0] in names}
        commands = command([str(Path(sys.executable).with_name("ninja")), "-C", str(build_cache),
                            "-t", "commands", "_core"])
        build["compile_commands"] = [line for line in commands.splitlines()
                                     if "/src/logistic.cpp" in line or "/bindings/python.cpp" in line]
    return dict(platform=platform.platform(), cpu=cpu, python=platform.python_version(),
                numpy=np.__version__, scipy=scipy.__version__, logitlab=version("logitlab"),
                compiled_module_sha256=hashlib.sha256(Path(_core.__file__).read_bytes()).hexdigest(),
                compiler=command(["g++", "--version"]),
                commit=command(["git", "rev-parse", "HEAD"]),
                source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                               for name in ("src/logistic.cpp", "include/logitlab/logistic_regression.hpp",
                                            "bindings/python.cpp", "python/logitlab/__init__.py",
                                            "benchmarks/suite.py", "CMakeLists.txt", "pyproject.toml")},
                heart_file_sha256=hashlib.sha256((ROOT / "data/heart_scale").read_bytes()).hexdigest(),
                working_tree=command(["git", "status", "--porcelain"]),
                cpu_affinity=sorted(os.sched_getaffinity(0)),
                memory_total=next(line for line in Path("/proc/meminfo").read_text().splitlines()
                                  if line.startswith("MemTotal:")),
                threads={name: os.environ[name] for name in os.environ if name.endswith("NUM_THREADS")},
                numpy_build_config=blas, build=build)


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = machine_metadata(args.build_cache)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    result = dict(schema_version=1, config=config,
                  machine=metadata, references=[], calibration=[], results=[])
    traces = []
    for case in args.cases:
        for seed in args.seeds:
            for scaling in ("raw", "rms"):
                start = time.perf_counter()
                xs, ys, meta = prepare(case, seed, scaling, args.quick)
                preparation_seconds = time.perf_counter() - start
                for l2 in args.l2:
                    ref = reference(xs[0], ys[0], l2)
                    keys = dict(case=case, seed=seed, scaling=scaling, l2=l2)
                    result["references"].append(keys | ref | meta)
                    # Interleave solver order deterministically to reduce order bias.
                    for solver in np.random.default_rng(seed).permutation(SOLVERS):
                        options = dict(solver=str(solver), max_epochs=args.epochs,
                                       batch_size=args.batch_size, tolerance=1e-6, seed=seed, l2=l2)
                        variants = [("baseline", options)]
                        if args.ablations and scaling == "rms":
                            if case == "sparse_wide" and solver in ("sgd", "adam") and l2 == .01:
                                variants += [(f"batch_{size}", options | {"batch_size": size})
                                             for size in (32, 1024) if size != args.batch_size]
                            if case == "dense_correlated" and solver == "lbfgs":
                                variants += [(f"history_{size}", options | {"history_size": size})
                                             for size in (3, 30)]
                        for variant, variant_options in variants:
                            evaluate_job(args, xs, ys, result, traces, keys | {"variant": variant},
                                         variant_options, ref, preparation_seconds)
    flattened = []
    for row in result["results"]:
        flat = {key: value for key, value in row.items() if not isinstance(value, dict)}
        for group in ("test", "validation", "target", "options"):
            flat.update({f"{group}_{key}": value for key, value in (row.get(group) or {}).items()})
        flattened.append(flat)
    write_csv(args.output / "metrics.csv", flattened)
    return result


def evaluate_job(args, xs, ys, result, traces, keys, options, ref, preparation_seconds):
    solver = options["solver"]
    selected, trials = calibrate(xs[0], ys[0], options)
    calibration = keys | dict(solver=solver, selected_options=selected,
                              trials=trials, seconds=sum(t["fit_seconds"] for t in trials))
    result["calibration"].append(calibration)
    job = keys | dict(quick=args.quick, options=selected, repeats=args.repeats,
                      reference=ref, target_gap=args.target_gap)
    child = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker"],
                           input=json.dumps(job), capture_output=True, text=True)
    if child.returncode:
        raise RuntimeError(f"worker failed for {keys}, {solver}: {child.stderr}")
    output = json.loads(child.stdout)
    for row in output["rows"]:
        row.update(keys, solver=solver, options=selected,
                   calibration_seconds=calibration["seconds"],
                   preparation_seconds=preparation_seconds)
    result["results"].extend(output["rows"])
    traces.extend(keys | dict(solver=solver) | h for h in output["traces"])
    print(f"{keys['case']} seed={keys['seed']} {keys['scaling']} l2={keys['l2']:g} "
          f"{solver} {keys['variant']}: "
          f"{output['rows'][0]['status']} ({output['rows'][0]['fit_seconds']:.3f}s)", flush=True)
    # Checkpoint completed jobs; an interrupted run retains its evidence.
    (args.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_csv(args.output / "traces.csv", traces)


def main():
    if sys.argv[1:] == ["--worker"]:
        print(json.dumps(worker(json.load(sys.stdin)), allow_nan=False))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Tiny protocol check, not performance evidence")
    parser.add_argument("--ablations", action=argparse.BooleanOptionalAction, default=True,
                        help="Also compare sparse mini-batch sizes and correlated L-BFGS history sizes")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--l2", type=float, nargs="+", default=[0., .01])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--target-gap", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, default=Path("build/benchmark-suite"))
    parser.add_argument("--build-cache", type=Path, help="Optional installed-wheel CMake cache for actual build flags")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.repeats) < 1 or args.target_gap <= 0:
        parser.error("budgets and target gap must be positive")
    if any(not np.isfinite(l2) or l2 < 0 for l2 in args.l2) or not np.isfinite(args.target_gap):
        parser.error("L2 must be finite/nonnegative and target gap finite/positive")
    if any(seed < 0 or seed >= 2**63 for seed in args.seeds):
        parser.error("seeds must be in [0, 2**63)")
    run(args)


if __name__ == "__main__":
    main()
