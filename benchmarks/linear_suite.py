"""Linear model/optimizer study; validation selects penalties, test stays held out."""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import warnings

import numpy as np
from scipy import sparse
from scipy.optimize import lsq_linear
from scipy.sparse.linalg import lsmr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks import suite
from logitlab import LinearRegression, regression_metrics

CASES = ("dense_control", "dense_correlated", "dense_multiscale",
         "sparse_wide", "heteroscedastic", "positive_prior")
PENALTIES = (.0001, .001, .01, .1)
MODELS = ("ols", "ridge", "lasso", "elastic-net")
SOLVERS = ("cg", "proximal-gd", "fista")


def dataset(case, seed, quick=False):
    rng = np.random.default_rng(seed)
    n, p = (120, 8) if quick else (5000, 40)
    if case == "sparse_wide":
        n, p = (180, 20) if quick else (6000, 1000)
        X = sparse.random(n, p, density=.15 if quick else .01, format="csr",
                          random_state=rng, data_rvs=lambda k: rng.normal(size=k))
    else:
        if not quick:
            if case == "dense_correlated": n, p = 2400, 48
            if case == "heteroscedastic": n, p = 1800, 24
            if case == "positive_prior": n, p = 500, 40
        X = rng.normal(size=(n, p))
        if case == "dense_correlated":
            latent = rng.normal(size=(n, p // 4))
            X = np.repeat(latent, 4, axis=1) + .05 * X
    truth = np.zeros(p)
    support = rng.choice(p, size=max(2, p // (30 if case == "sparse_wide" else 5)), replace=False)
    truth[support] = rng.normal(size=len(support)) * 2
    intercept = 2.
    if case == "dense_multiscale":
        factors = np.geomspace(.01, 100, p)
        X *= factors
        truth /= factors
    if case == "positive_prior":
        # Declared domain knowledge, fixed before generating responses.
        X *= np.geomspace(.1, 10, p)
        truth[:] = 0
        truth[support] = rng.uniform(.1, .5, size=len(support))
        truth[0] = .2
    sigma = (.15 + 3 * np.abs(X[:, 0]) if case == "heteroscedastic"
             else np.full(n, 1.5 if case == "positive_prior" else 1.))
    precision = 1 / sigma**2
    y = np.asarray(X @ truth).ravel() + intercept + rng.normal(size=n) * sigma
    return X, y, precision, truth


def prepare(case, seed, scaling, quick=False):
    X, y, weights, truth = dataset(case, seed, quick)
    indices = np.random.default_rng(seed + 200000).permutation(len(y))
    a, b = int(.7 * len(y)), int(.85 * len(y))
    splits = (indices[:a], indices[a:b], indices[b:])
    xs, ys, ws = [X[s] for s in splits], [y[s] for s in splits], [weights[s] for s in splits]
    train_scale = suite.rms_scale(xs[0])
    arrays = (X.data, X.indices, X.indptr, y, weights, truth) if sparse.issparse(X) else (X, y, weights, truth)
    meta = dict(rows=X.shape[0], cols=X.shape[1], train_rows=a, validation_rows=b-a, test_rows=len(y)-b,
                storage="csr" if sparse.issparse(X) else "dense",
                train_nnz=int(xs[0].nnz) if sparse.issparse(X) else int(xs[0].size),
                data_sha256=suite.digest(*arrays), split_sha256=suite.digest(*splits),
                scale_sha256=suite.digest(train_scale))
    scale = train_scale if scaling == "rms" else np.ones(X.shape[1])
    if scaling == "rms":
        xs = [suite.divide(x, scale) for x in xs]
    return xs, ys, ws, scale, truth, meta


def models(case):
    return list(MODELS) + (["wls"] if case == "heteroscedastic" else
                          ["nonnegative", "box-fixed"] if case == "positive_prior" else [])


def model_options(name, alpha, scale):
    l1 = alpha if name == "lasso" else .5 * alpha if name == "elastic-net" else 0.
    l2 = alpha if name == "ridge" else .5 * alpha if name == "elastic-net" else 0.
    options = dict(l1=l1, l2=l2)
    if name == "nonnegative":
        options["positive"] = True
    if name == "box-fixed":
        options.update(bounds=(0., .5 * scale), fixed_coefficients={0: .2 * scale[0]})
    return options


def bounds(options, p):
    lo, hi = options.get("bounds", (None, None))
    lower = np.broadcast_to(-np.inf if lo is None else lo, (p,)).astype(float).copy()
    upper = np.broadcast_to(np.inf if hi is None else hi, (p,)).astype(float).copy()
    if options.get("positive", False):
        lower = np.maximum(lower, 0)
    for key, value in options.get("fixed_coefficients", {}).items():
        lower[int(key)] = upper[int(key)] = value
    return np.r_[lower, -np.inf], np.r_[upper, np.inf]


def objective(X, y, theta, options, sample_weight=None):
    q = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight) / np.max(sample_weight)
    q = q / q.sum()
    residual = np.asarray(X @ theta[:-1]).ravel() + theta[-1] - y
    l1, l2 = options["l1"], options["l2"]
    loss = .5 * np.dot(q, residual**2) + l1 * np.abs(theta[:-1]).sum() + .5 * l2 * np.dot(theta[:-1], theta[:-1])
    gradient = np.r_[np.asarray(X.T @ (q * residual)).ravel() + l2 * theta[:-1], q @ residual]
    lower, upper = bounds(options, X.shape[1])
    proposal = theta - gradient
    proposal[:-1] = np.sign(proposal[:-1]) * np.maximum(np.abs(proposal[:-1]) - l1, 0)
    mapping = theta - np.clip(proposal, lower, upper)
    violation = max(float(np.maximum(lower-theta, 0).max()), float(np.maximum(theta-upper, 0).max()))
    return float(loss), float(np.linalg.norm(mapping)), violation


def reference(X, y, options, sample_weight=None):
    start = time.perf_counter()
    p = X.shape[1]
    q = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight) / np.max(sample_weight)
    q = q / q.sum()
    messages = []
    if options["l1"] > 0:
        from sklearn.linear_model import ElasticNet
        from sklearn.exceptions import ConvergenceWarning
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            reg = ElasticNet(alpha=options["l1"] + options["l2"],
                             l1_ratio=options["l1"] / (options["l1"] + options["l2"]),
                             fit_intercept=True, tol=1e-11, max_iter=30000,
                             selection="cyclic").fit(X, y, sample_weight=q * len(y))
        theta = np.r_[reg.coef_, reg.intercept_]
        messages = [str(w.message) for w in caught]
        method = "sklearn cyclic coordinate descent"
    else:
        # Column precondition a separately formulated weighted least-squares
        # system. Penalty and bounds transform too, preserving the objective.
        A = sparse.hstack((X, np.ones((len(y), 1))), format="csr") if sparse.issparse(X) else np.c_[X, np.ones(len(y))]
        diag = np.asarray(A.power(2).T @ q).ravel() if sparse.issparse(A) else q @ (A**2)
        scale = np.sqrt(diag); scale[scale == 0] = 1.
        Z = suite.divide(A, scale)
        R = np.sqrt(np.r_[np.full(p, options["l2"]), 0.]) / scale
        if sparse.issparse(A):
            B = sparse.vstack((Z.multiply(np.sqrt(q)[:, None]), sparse.diags(R)), format="csr")
        else:
            B = np.vstack((Z * np.sqrt(q[:, None]), np.diag(R)))
        response = np.r_[y * np.sqrt(q), np.zeros(p + 1)]
        lower, upper = bounds(options, p)
        lower, upper = lower * scale, upper * scale
        fixed = lower == upper; free = ~fixed
        transformed = np.zeros(p+1); transformed[fixed] = lower[fixed]
        response = response - np.asarray(B[:, fixed] @ transformed[fixed]).ravel()
        if np.isfinite(lower[free]).any() or np.isfinite(upper[free]).any():
            result = lsq_linear(B[:, free], response, bounds=(lower[free], upper[free]),
                                tol=1e-12, max_iter=3000, lsmr_tol=1e-12)
            transformed[free] = result.x
            messages = [str(result.message)]
            method = "scipy bounded least squares with fixed coordinates eliminated"
        elif sparse.issparse(B):
            result = lsmr(B, response, atol=1e-12, btol=1e-12, maxiter=10000)
            transformed = result[0]
            messages = [f"lsmr istop={result[1]}, iterations={result[2]}"]
            method = "scipy sparse LSMR"
        else:
            transformed = np.linalg.lstsq(B, response, rcond=None)[0]
            method = "NumPy SVD least squares"
        theta = transformed / scale
    loss, stationarity, violation = objective(X, y, theta, options, sample_weight)
    return dict(theta=theta, objective=loss, stationarity_norm=stationarity,
                constraint_violation=violation, verified=stationarity <= 1e-5 and violation <= 1e-10,
                method=method, messages=messages, seconds=time.perf_counter()-start)


def select_model(name, xs, ys, scale, sample_weight):
    trials, selected = [], None
    for alpha in PENALTIES if name in ("ridge", "lasso", "elastic-net") else (0.,):
        options = model_options(name, alpha, scale)
        ref = reference(xs[0], ys[0], options, sample_weight)
        prediction = np.asarray(xs[1] @ ref["theta"][:-1]).ravel() + ref["theta"][-1]
        validation = regression_metrics(ys[1], prediction)
        trial = dict(alpha=alpha, validation=validation, **{k: v for k, v in ref.items() if k != "theta"})
        trials.append(trial)
        if ref["verified"] and (selected is None or validation["mse"] < selected[0]):
            selected = (validation["mse"], alpha, options, ref)
    if selected is None:
        raise RuntimeError(f"No verified reference for {name}: {trials}")
    return selected[1:], trials


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def worker(job):
    xs, ys, weights, scale, truth, meta = prepare(job["case"], job["seed"], job["scaling"], job["quick"])
    options = job["options"]
    if "fixed_coefficients" in options:
        options["fixed_coefficients"] = {int(k): v for k, v in options["fixed_coefficients"].items()}
    weighted = weights[0] if job["model"] == "wls" else None
    LinearRegression(**(options | {"max_iter": 1})).fit(
        xs[0][:32], ys[0][:32], sample_weight=weighted[:32] if weighted is not None else None)
    rows, traces = [], []
    for repeat in range(job["repeats"]):
        start = time.perf_counter()
        model = LinearRegression(**options).fit(xs[0], ys[0], sample_weight=weighted)
        elapsed = time.perf_counter() - start
        model.predict(xs[2])
        durations = []
        for _ in range(7):
            start = time.perf_counter()
            prediction = model.predict(xs[2])
            durations.append(time.perf_counter() - start)
        theta = np.r_[model.coef_, model.intercept_]
        loss, stationarity, violation = objective(xs[0], ys[0], theta, options, weighted)
        ref = job["reference"]
        valid = ref["verified"] and loss >= ref["objective"] - 1e-7 * (1 + abs(ref["objective"]))
        gap = max(0., (loss - ref["objective"]) / (1 + abs(ref["objective"]))) if valid else None
        target = None
        if valid:
            for history in model.history_:
                if (history["loss"] - ref["objective"]) / (1 + abs(ref["objective"])) <= job["target_gap"]:
                    target = {key: history[key] for key in ("iteration", "seconds", "row_visits")}
                    break
        recovered = model.coef_ / scale
        selected, actual = abs(recovered) > 1e-6, abs(truth) > 1e-6
        tp, fp, fn = np.sum(selected & actual), np.sum(selected & ~actual), np.sum(~selected & actual)
        support_f1 = float(2*tp / (2*tp+fp+fn)) if 2*tp+fp+fn else 1.
        rows.append(dict(repeat=repeat, **meta, status=model.status_, converged=model.converged_,
                         iterations=model.history_[-1]["iteration"], fit_seconds=elapsed,
                         cpp_seconds=model.history_[-1]["seconds"], **model.work_,
                         train_objective=loss, stationarity_norm=stationarity,
                         reference_valid=valid, objective_gap=gap, target=target,
                         constraint_violation=violation, nonzero_coefficients=int(selected.sum()),
                         coefficient_rmse=float(np.sqrt(np.mean((recovered-truth)**2))),
                         support_f1=support_f1, prediction_rows_per_second=len(ys[2]) / np.median(durations),
                         reference_test=ref["test"],
                         test=regression_metrics(ys[2], prediction),
                         test_weighted=regression_metrics(ys[2], prediction, sample_weight=weights[2])))
        # Every repeat retains metrics and target timing; one full trace is enough
        # for deterministic parameter trajectories and avoids 3x repository size.
        if repeat == 0:
            traces.extend(dict(repeat=repeat, **h) for h in model.history_)
        del model
    peak = int(next(line.split()[1] for line in Path("/proc/self/status").read_text().splitlines()
                    if line.startswith("VmHWM:"))) / 1024
    for row in rows: row["worker_peak_rss_mib"] = peak
    return dict(rows=rows, traces=traces)


def run(args):
    # Reference calibration timing excludes one-time sklearn imports.
    import sklearn.linear_model
    args.output.mkdir(parents=True, exist_ok=True)
    machine = suite.machine_metadata(args.build_cache)
    from importlib.metadata import version
    machine["sklearn"] = version("scikit-learn")
    for name in ("src/linear.cpp", "include/logitlab/linear_regression.hpp", "python/logitlab/linear.py",
                 "python/logitlab/analysis.py", "benchmarks/linear_suite.py"):
        machine["source_sha256"][name] = hashlib.sha256((suite.ROOT/name).read_bytes()).hexdigest()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    data = dict(schema_version=1, config=config, machine=machine, selections=[], results=[])
    traces = []
    for case in args.cases:
        for seed in args.seeds:
            for scaling in ("raw", "rms"):
                xs, ys, weights, scale, truth, meta = prepare(case, seed, scaling, args.quick)
                for name in models(case):
                    selected, trials = select_model(name, xs, ys, scale, weights[0] if name == "wls" else None)
                    alpha, base, ref = selected
                    ref["test"] = regression_metrics(
                        ys[2], np.asarray(xs[2] @ ref["theta"][:-1]).ravel() + ref["theta"][-1])
                    ref = {k: v for k, v in ref.items() if k != "theta"}
                    keys = dict(case=case, seed=seed, scaling=scaling, model=name)
                    selection = keys | dict(alpha=alpha, options=base, reference=ref, trials=trials,
                                             selection_seconds=sum(trial["seconds"] for trial in trials), **meta)
                    data["selections"].append(jsonable(selection))
                    solvers = SOLVERS if name in ("ols", "ridge", "wls") else ("proximal-gd", "fista")
                    for solver in np.random.default_rng(seed).permutation(solvers):
                        options = base | dict(solver=str(solver), max_iter=args.max_iter, tolerance=1e-7)
                        job = keys | dict(quick=args.quick, options=options, repeats=args.repeats,
                                          reference=ref, target_gap=args.target_gap)
                        child = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker"],
                                               input=json.dumps(jsonable(job)), capture_output=True, text=True)
                        if child.returncode:
                            raise RuntimeError(f"worker {keys}/{solver}: {child.stderr}")
                        result = json.loads(child.stdout)
                        for row in result["rows"]:
                            row.update(keys, solver=str(solver), alpha=alpha,
                                       selection_seconds=selection["selection_seconds"])
                        data["results"].extend(result["rows"])
                        traces.extend(keys | dict(solver=str(solver)) | h for h in result["traces"])
                        print(f"{case} seed={seed} {scaling} {name}/{solver}: "
                              f"{result['rows'][0]['status']}, alpha={alpha:g}", flush=True)
                        (args.output/"results.json").write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
                        suite.write_csv(args.output/"traces.csv", traces)
    flat = []
    for row in data["results"]:
        out = {k: v for k, v in row.items() if not isinstance(v, dict)}
        for key in ("test", "test_weighted", "reference_test", "target"):
            out.update({f"{key}_{k}": v for k, v in (row.get(key) or {}).items()})
        flat.append(out)
    suite.write_csv(args.output/"metrics.csv", flat)
    return data


def main():
    if sys.argv[1:] == ["--worker"]:
        print(json.dumps(worker(json.load(sys.stdin)), allow_nan=False))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--cases", choices=CASES, nargs="+", default=list(CASES))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--target-gap", type=float, default=1e-5)
    parser.add_argument("--build-cache", type=Path)
    parser.add_argument("--output", type=Path, default=Path("build/linear-benchmark"))
    args = parser.parse_args()
    if min(args.max_iter, args.repeats) < 1 or not np.isfinite(args.target_gap) or args.target_gap <= 0:
        parser.error("positive finite budgets and gap required")
    if any(seed < 0 or seed >= 2**63 for seed in args.seeds) or len(args.seeds) != len(set(args.seeds)):
        parser.error("unique seeds in [0, 2**63) required")
    if len(args.cases) != len(set(args.cases)):
        parser.error("cases must be unique")
    run(args)


if __name__ == "__main__":
    main()
