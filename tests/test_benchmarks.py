"""Scientific protocol invariants and actual core work accounting."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

# Research tools live in the checkout, including when wheel tests run from /tmp.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks import report, suite
from logitlab import LogisticRegression


@pytest.mark.parametrize("case", suite.CASES)
def test_reproducible_splits_and_scaling(case):
    xs, ys, meta = suite.prepare(case, 42, "raw", True)
    scaled, labels, scaled_meta = suite.prepare(case, 42, "rms", True)
    assert meta == scaled_meta
    X, y = suite.dataset(case, 42, True)
    splits = suite.split_indices(y, 42)
    assert len(set(np.concatenate(splits))) == len(y)
    assert all(len(np.unique(part)) == 2 for part in ys)
    scale = suite.rms_scale(xs[0])
    for raw, normalized, a, b in zip(xs, scaled, ys, labels):
        expected = suite.divide(raw, scale)
        dense = lambda x: x.toarray() if sparse.issparse(x) else x
        np.testing.assert_allclose(dense(expected), dense(normalized))
        np.testing.assert_array_equal(a, b)
    if case == "sparse_wide":
        assert sparse.isspmatrix_csr(scaled[0])
        assert scaled[0].nnz == xs[0].nnz
    np.testing.assert_allclose(suite.rms_scale(scaled[0]), 1., atol=1e-14)


def test_reference_preserves_raw_regularizer_and_gradient():
    X, y = suite.dataset("dense_multiscale", 42, True)
    scale = suite.rms_scale(X)
    theta = np.linspace(-.1, .2, X.shape[1] + 1)
    raw = np.r_[theta[:-1] / scale, theta[-1]]
    f, gradient = suite.objective(suite.divide(X, scale), y, theta, .01 / scale**2)
    raw_f, raw_gradient = suite.objective(X, y, raw, np.full(X.shape[1], .01))
    assert f == pytest.approx(raw_f)
    np.testing.assert_allclose(gradient, np.r_[raw_gradient[:-1] / scale, raw_gradient[-1]])
    for j in range(len(theta)):
        delta = np.eye(len(theta))[j] * 1e-6
        hi = suite.objective(suite.divide(X, scale), y, theta + delta, .01 / scale**2)[0]
        lo = suite.objective(suite.divide(X, scale), y, theta - delta, .01 / scale**2)[0]
        assert gradient[j] == pytest.approx((hi - lo) / 2e-6, abs=1e-6)


def test_metrics_ties_and_censored_targets():
    y = np.array([0., 1., 0., 1.])
    m = suite.metrics(y, np.full(4, .5))
    assert m == pytest.approx(dict(log_loss=np.log(2), accuracy=.5, roc_auc=.5, brier=.25))
    assert suite.metrics(y, y)["roc_auc"] == 1
    assert suite.metrics(y, 1 - y)["roc_auc"] == 0
    assert suite.metrics(np.ones(3), np.ones(3))["roc_auc"] is None
    ref = dict(verified=True, objective=.4)
    history = [dict(epoch=0, loss=.69, seconds=.1, row_visits=10),
               dict(epoch=1, loss=.40001, seconds=.2, row_visits=20)]
    assert suite.first_target(history, ref, 1e-4)["epoch"] == 1
    assert suite.first_target(history[:1], ref, 1e-4) is None
    assert suite.first_target(history, ref | {"verified": False}, 1e-4) is None
    assert suite.first_target(history, ref | {"objective": .5}, 1e-4) is None


@pytest.mark.parametrize("solver", suite.SOLVERS)
def test_core_work_counters(solver):
    X = np.array([[-2.], [-1.], [1.], [2.], [3.]])
    y = np.array([0., 0., 1., 1., 0.])
    model = LogisticRegression(solver=solver, max_epochs=1, batch_size=2, l2=.1).fit(X, y)
    first, last, total = model.history_[0], model.history_[-1], model.work_
    assert first["full_evaluations"] == 1
    assert first["row_visits"] == len(y)
    assert all(last[key] == total[key] for key in total)
    if solver in ("sgd", "adam"):
        assert total == dict(full_evaluations=2, batch_gradients=3,
                            hessian_vector_products=0, row_visits=15, parameter_updates=3)
    else:
        assert total["parameter_updates"] == 1
        assert total["batch_gradients"] == 0
        extra = total["hessian_vector_products"] + (solver == "newton-cg")
        assert total["row_visits"] == (total["full_evaluations"] + extra) * len(y)
        assert (total["hessian_vector_products"] > 0) == (solver == "newton-cg")


def test_failed_line_search_still_counts_work():
    X, y = np.array([[-1.], [1.]]), np.array([0., 1.])
    model = LogisticRegression(solver="gd", learning_rate=1e300, max_epochs=1).fit(X, y)
    assert model.status_ == "line_search_failed"
    assert model.work_["full_evaluations"] == 41
    assert model.work_["row_visits"] == 82
    assert model.history_[-1]["full_evaluations"] == 1


def test_summary_uses_seeds_and_preserves_unreached():
    base = dict(case="dense_control", l2=0., scaling="rms", solver="sgd", variant="baseline",
                status="max_epochs", objective_gap=.1, reference_valid=True,
                train_rows=10, row_visits=30, target=None)
    rows = [base | dict(seed=42, repeat=i, fit_seconds=1.) for i in range(3)]
    rows += [base | dict(seed=43, repeat=0, fit_seconds=100.)]
    summary, = report.summarize(rows, 1e-4)
    assert summary["fit_seconds"]["median"] == 50.5  # NOT median of all repeats.
    assert summary["seeds"] == 2 and summary["repeats"] == 4
    assert summary["target_seconds"] is None
    assert summary["ever_target_seeds"] == summary["final_target_seeds"] == 0
    assert report.fmt(summary, "target_seconds") == "—"
    rows[0]["target"] = dict(seconds=.1, row_visits=20)
    summary, = report.summarize(rows, 1e-4)
    assert summary["ever_target_seeds"] == 1 and summary["final_target_seeds"] == 0
    assert summary["target_seconds"]["n"] == 1
    assert summary["target_passes"]["median"] == 2


def test_worker_and_cli_protocol(tmp_path):
    script = Path(suite.__file__)
    child = subprocess.run([sys.executable, str(script), "--quick", "--cases", "dense_control",
                            "--seeds", "42", "--l2", ".01", "--epochs", "2", "--repeats", "1",
                            "--output", str(tmp_path)], capture_output=True, text=True, check=True)
    data = json.loads((tmp_path / "results.json").read_text())
    report.validate_complete(data)
    assert len(data["results"]) == 10
    assert len(data["calibration"]) == 10
    assert all(ref["verified"] for ref in data["references"])
    assert {row["solver"] for row in data["results"]} == set(suite.SOLVERS)
    assert all(row["worker_peak_rss_mib"] > 0 for row in data["results"])
    assert all(row["test"]["log_loss"] >= 0 for row in data["results"])
    assert (tmp_path / "metrics.csv").is_file() and (tmp_path / "traces.csv").is_file()
    assert "dense_control" in child.stdout
    summaries = report.summarize(data["results"], 1e-4)
    for lang in ("en", "zh-CN"):
        rendered = report.render(data, summaries, lang, "../raw")
        assert "10" in rendered
        assert "../raw/results.json" in rendered
        assert all(solver in rendered for solver in suite.SOLVERS)
    with pytest.raises(ValueError, match="missing measurement slots"):
        report.validate_complete(data | {"results": data["results"][:-1]})


@pytest.mark.skipif(sys.platform != "linux", reason="VmHWM is Linux-specific")
def test_worker_memory_excludes_launcher_high_water():
    xs, ys, _ = suite.prepare("dense_control", 42, "rms", True)
    job = dict(case="dense_control", seed=42, scaling="rms", quick=True, repeats=1,
               options=dict(solver="lbfgs", max_epochs=2, l2=.01),
               reference=suite.reference(xs[0], ys[0], .01), target_gap=1e-4)
    # Old getrusage accounting inherited this allocation through fork/exec.
    launcher_buffer = np.ones(8_000_000, dtype=np.float64)
    parent_peak = int(next(line.split()[1] for line in Path("/proc/self/status").read_text().splitlines()
                           if line.startswith("VmHWM:"))) / 1024
    child = subprocess.run([sys.executable, suite.__file__, "--worker"], input=json.dumps(job),
                           capture_output=True, text=True, check=True)
    worker_peak = json.loads(child.stdout)["rows"][0]["worker_peak_rss_mib"]
    assert launcher_buffer.nbytes == 64_000_000
    assert worker_peak < parent_peak - 16
