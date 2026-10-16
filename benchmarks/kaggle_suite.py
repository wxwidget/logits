"""Held-out Kaggle study of logits optimizers and linear regularization.

Download first with kaggle_data.py. This runner never downloads implicitly.
"""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks import kaggle_data, linear_suite, suite
from logitlab import LinearRegression, LogisticRegression, regression_metrics

LINEAR_MODELS = ("ols", "ridge", "lasso", "elastic-net")
ALPHAS = (.0001, .01, .1)


def load_prepared(case, cache, seed, scaling, allow_fixtures=False):
    start = time.perf_counter()
    rows, y, manifest = kaggle_data.load(case, cache)
    if manifest.get("evidence_kind") != "kaggle-download" and not allow_fixtures:
        raise ValueError("Refusing fixture/unverified data as real Kaggle evidence")
    prepared = kaggle_data.prepare(case, rows, y, seed, scaling)
    prepared.update(manifest=manifest, preparation_seconds=time.perf_counter()-start)
    return prepared


def classification_metrics(y, p):
    from sklearn.metrics import average_precision_score
    return suite.metrics(y, p) | dict(average_precision=float(average_precision_score(y, p)))


def logistic_reference(X, y, l2):
    """Independent SciPy solution with an exactly transformed L2 penalty."""
    scale = suite.rms_scale(X)
    Z = suite.divide(X, scale)
    penalty = l2 / scale**2
    fun = lambda theta: suite.objective(Z, y, theta, penalty)
    def hvp(theta, vector):
        p = expit(np.asarray(Z @ theta[:-1]).ravel() + theta[-1])
        u = np.asarray(Z @ vector[:-1]).ravel() + vector[-1]
        r = p * (1-p) * u / len(y)
        return np.r_[np.asarray(Z.T @ r).ravel() + penalty * vector[:-1], r.sum()]
    start = time.perf_counter()
    first = minimize(fun, np.zeros(X.shape[1]+1), jac=True, hessp=hvp, method="Newton-CG",
                     options=dict(xtol=1e-10, maxiter=500))
    result = first
    if np.linalg.norm(fun(first.x)[1]) > 1e-7:
        second = minimize(fun, first.x, jac=True, method="L-BFGS-B",
                          options=dict(ftol=1e-14, gtol=1e-9, maxiter=5000, maxls=40))
        if second.fun < first.fun:
            result = second
    loss, gradient = fun(result.x)
    norm = float(np.linalg.norm(gradient))
    return dict(theta=np.r_[result.x[:-1]/scale, result.x[-1]], objective=loss,
                gradient_norm_preconditioned=norm, verified=norm <= 1e-5,
                method="SciPy Newton-CG with L-BFGS-B fallback", message=str(result.message),
                seconds=time.perf_counter()-start)


def select_model(task, name, xs, ys, scale):
    """No test targets are accessed during selection."""
    trials, selected = [], None
    for alpha in (0.,) if name == "ols" else ALPHAS:
        options = (dict(l2=alpha) if task == "classification"
                   else linear_suite.model_options(name, alpha, scale))
        ref = (logistic_reference(xs[0], ys[0], alpha) if task == "classification"
               else linear_suite.reference(xs[0], ys[0], options))
        prediction = np.asarray(xs[1] @ ref["theta"][:-1]).ravel() + ref["theta"][-1]
        metrics = (classification_metrics(ys[1], expit(prediction)) if task == "classification"
                   else regression_metrics(ys[1], prediction))
        score = metrics["log_loss" if task == "classification" else "mse"]
        trials.append(dict(alpha=alpha, validation=metrics, **{k:v for k,v in ref.items() if k != "theta"}))
        if ref["verified"] and (selected is None or score < selected[0]):
            selected = score, alpha, options, ref
    if selected is None:
        raise RuntimeError(f"No verified reference for {name}; inspect trial evidence: {trials}")
    return selected[1:], trials


def solvers(name):
    if name == "logistic":
        return suite.SOLVERS
    return linear_suite.SOLVERS if name in ("ols", "ridge") else ("proximal-gd", "fista")


def names(case):
    return ("logistic",) if kaggle_data.CATALOG[case]["task"] == "classification" else LINEAR_MODELS


def prediction_metrics(prepared, part, prediction, classification):
    y = prepared["ys"][part]
    if classification:
        return classification_metrics(y, prediction)
    shift, scale = prepared["target_center"], prepared["target_scale"]
    return regression_metrics(y * scale + shift, prediction * scale + shift)


def worker(job):
    prepared = load_prepared(job["case"], job["cache"], job["seed"], job["scaling"], job["allow_fixtures"])
    xs, ys = prepared["xs"], prepared["ys"]
    classification = job["model"] == "logistic"
    klass = LogisticRegression if classification else LinearRegression
    options = job["options"]
    budget_key = "max_epochs" if classification else "max_iter"
    klass(**(options | {budget_key: 1})).fit(xs[0][:32], ys[0][:32])
    rows, traces = [], []
    for repeat in range(job["repeats"]):
        start = time.perf_counter()
        try:
            model = klass(**options).fit(xs[0], ys[0])
        except RuntimeError as error:
            rows.append(dict(repeat=repeat, fit_seconds=time.perf_counter()-start, status="error",
                             error=str(error), reference_valid=False, target=None, test=None))
            continue
        elapsed = time.perf_counter()-start
        predict = (lambda: model.predict_proba(xs[2])[:, 1]) if classification else (lambda: model.predict(xs[2]))
        predict()
        timings = []
        for _ in range(7):
            start = time.perf_counter(); prediction = predict(); timings.append(time.perf_counter()-start)
        theta = np.r_[model.coef_, model.intercept_]
        if classification:
            loss, gradient = suite.objective(xs[0], ys[0], theta, options["l2"])
            stationarity = float(np.linalg.norm(gradient))
        else:
            loss, stationarity, _ = linear_suite.objective(xs[0], ys[0], theta, options)
        ref = job["reference"]
        valid = ref["verified"] and loss >= ref["objective"] - 1e-7 * (1+abs(ref["objective"]))
        gap = max(0., (loss-ref["objective"])/(1+abs(ref["objective"]))) if valid else None
        target = None
        if valid:
            for h in model.history_:
                if (h["loss"]-ref["objective"])/(1+abs(ref["objective"])) <= job["target_gap"]:
                    target = {k:h[k] for k in ("seconds", "row_visits")}
                    break
        rows.append(dict(repeat=repeat, fit_seconds=elapsed, cpp_seconds=model.history_[-1]["seconds"],
                         status=model.status_, converged=model.converged_, error=None,
                         iterations=model.history_[-1]["epoch" if classification else "iteration"],
                         train_objective=loss, stationarity_norm=stationarity, objective_gap=gap,
                         reference_valid=valid, target=target, **model.work_,
                         nonzero_coefficients=int((abs(model.coef_) > 1e-6).sum()),
                         coefficients_original_units=(model.coef_ / prepared["scale"] *
                                                      prepared["target_scale"]).tolist(),
                         intercept_original_units=float(model.intercept_*prepared["target_scale"] + prepared["target_center"]),
                         prediction_rows_per_second=len(ys[2])/float(np.median(timings)),
                         test=prediction_metrics(prepared, 2, prediction, classification)))
        if repeat == 0:
            traces.extend(dict(repeat=repeat, **h) for h in model.history_)
    peak = int(next(line.split()[1] for line in Path("/proc/self/status").read_text().splitlines()
                    if line.startswith("VmHWM:")))/1024
    for row in rows:
        row.update(prepared["meta"], preparation_seconds=prepared["preparation_seconds"], worker_peak_rss_mib=peak)
    return dict(rows=rows, traces=traces)


def run(args):
    import sklearn.linear_model
    args.output.mkdir(parents=True, exist_ok=True)
    machine = suite.machine_metadata(args.build_cache)
    machine["sklearn"] = version("scikit-learn")
    for filename in ("benchmarks/kaggle_data.py", "benchmarks/kaggle_suite.py", "benchmarks/linear_suite.py",
                     "src/linear.cpp", "include/logitlab/linear_regression.hpp", "python/logitlab/linear.py",
                     "python/logitlab/analysis.py"):
        machine["source_sha256"][filename] = hashlib.sha256((suite.ROOT/filename).read_bytes()).hexdigest()
    config = {k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    data = dict(schema_version=1, config=config, machine=machine, evidence_kind="kaggle-download",
                datasets={}, selections=[], calibrations=[], results=[])
    traces = []
    for case in args.cases:
        for seed in args.seeds:
            for scaling in ("raw", "rms"):
                prepared = load_prepared(case, args.cache, seed, scaling, args.allow_fixtures)
                xs, ys, scale = prepared["xs"], prepared["ys"], prepared["scale"]
                data["datasets"][case] = prepared["manifest"]
                if prepared["manifest"]["evidence_kind"] != "kaggle-download":
                    data["evidence_kind"] = "protocol-fixture"
                classification = kaggle_data.CATALOG[case]["task"] == "classification"
                task = "classification" if classification else "regression"
                baseline = np.full(len(ys[2]), ys[0].mean())
                for name in names(case):
                    (alpha, base, ref), trials = select_model(task, name, xs, ys, scale)
                    prediction = np.asarray(xs[2] @ ref["theta"][:-1]).ravel() + ref["theta"][-1]
                    if classification:
                        prediction = expit(prediction)
                    reference_test = prediction_metrics(prepared, 2, prediction, classification)
                    ref = {k:v for k,v in ref.items() if k != "theta"}
                    keys = dict(case=case, seed=seed, scaling=scaling, model=name)
                    selection = keys | dict(alpha=alpha, reference=ref, reference_test=reference_test,
                                             baseline_test=prediction_metrics(prepared, 2, baseline, classification),
                                             trials=trials, selection_seconds=sum(t["seconds"] for t in trials),
                                             feature_names=prepared["feature_names"], **prepared["meta"])
                    data["selections"].append(selection)
                    for solver in np.random.default_rng(seed).permutation(solvers(name)):
                        solver = str(solver)
                        options = base | dict(solver=solver, tolerance=1e-7)
                        if classification:
                            options.update(max_epochs=args.iterations, seed=seed, batch_size=256)
                            options, calibration = suite.calibrate(xs[0], ys[0], options)
                            data["calibrations"].append(keys | dict(solver=solver, trials=calibration,
                                                                   selected_options=options))
                            calibration_seconds = sum(t["fit_seconds"] for t in calibration)
                        else:
                            options.update(max_iter=args.iterations)
                            calibration_seconds = 0.
                        job = keys | dict(cache=str(args.cache.resolve()), allow_fixtures=args.allow_fixtures,
                                          options=options, reference=ref, repeats=args.repeats, target_gap=args.target_gap)
                        child = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker"],
                                               input=json.dumps(job), capture_output=True, text=True)
                        if child.returncode:
                            raise RuntimeError(f"{keys}/{solver}: {child.stderr}")
                        result = json.loads(child.stdout)
                        for row in result["rows"]:
                            row.update(keys, solver=solver, alpha=alpha, options=options,
                                       calibration_seconds=calibration_seconds,
                                       selection_seconds=selection["selection_seconds"])
                        data["results"].extend(result["rows"])
                        traces.extend(keys | dict(solver=solver) | h for h in result["traces"])
                        print(f"{case}/{seed}/{scaling}/{name}/{solver}: "
                              f"{dict(Counter(r['status'] for r in result['rows']))}", flush=True)
                        (args.output/"results.json").write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
                        suite.write_csv(args.output/"traces.csv", traces)
    flat = []
    for row in data["results"]:
        out = {k:v for k,v in row.items() if not isinstance(v, dict)}
        for key in ("test", "target"):
            out.update({f"{key}_{k}":v for k,v in (row.get(key) or {}).items()})
        flat.append(out)
    suite.write_csv(args.output/"metrics.csv", flat)
    return data


def main():
    if sys.argv[1:] == ["--worker"]:
        print(json.dumps(worker(json.load(sys.stdin)), allow_nan=False)); return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=kaggle_data.CATALOG, default=list(kaggle_data.CATALOG))
    parser.add_argument("--cache", type=Path, default=kaggle_data.DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=Path("build/kaggle-benchmark"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--target-gap", type=float, default=1e-5)
    parser.add_argument("--build-cache", type=Path)
    parser.add_argument("--allow-fixtures", action="store_true", help="Protocol testing only; labels all output as fixture evidence")
    args = parser.parse_args()
    if min(args.repeats, args.iterations) < 1 or not np.isfinite(args.target_gap) or args.target_gap <= 0:
        parser.error("positive finite budgets and gap required")
    if len(args.cases) != len(set(args.cases)) or len(args.seeds) != len(set(args.seeds)):
        parser.error("cases/seeds must be unique")
    if any(seed < 0 or seed >= 2**63 for seed in args.seeds):
        parser.error("seed out of range")
    run(args)


if __name__ == "__main__":
    main()
