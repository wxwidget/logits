"""Benchmark protocol contracts before adding the regression runner."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks import linear_suite


@pytest.mark.parametrize("case", linear_suite.CASES)
def test_scenarios_splits_and_constraint_units(case):
    a = linear_suite.prepare(case, 42, "raw", True)
    b = linear_suite.prepare(case, 42, "rms", True)
    xs, ys, weights, scale, truth, meta = a
    scaled_xs, scaled_y, scaled_weights, fitted_scale, scaled_truth, scaled_meta = b
    assert meta == scaled_meta
    np.testing.assert_array_equal(truth, scaled_truth)
    for X, Z, y, zy, w, zw in zip(xs, scaled_xs, ys, scaled_y, weights, scaled_weights):
        left = X.toarray() if sparse.issparse(X) else X
        right = Z.toarray() if sparse.issparse(Z) else Z
        np.testing.assert_allclose(right, left / fitted_scale)
        np.testing.assert_array_equal(y, zy); np.testing.assert_array_equal(w, zw)
    np.testing.assert_array_equal(scale, np.ones_like(scale))
    if case == "sparse_wide":
        assert sparse.isspmatrix_csr(scaled_xs[0])
    options = linear_suite.model_options("box-fixed", 0., fitted_scale)
    assert options["fixed_coefficients"][0] == pytest.approx(.2 * fitted_scale[0])
    np.testing.assert_allclose(options["bounds"][1], .5 * fitted_scale)


def test_reference_and_objective_normalization():
    pytest.importorskip("sklearn")
    xs, ys, weights, scale, truth, meta = linear_suite.prepare("dense_control", 42, "rms", True)
    for name in ("ols", "ridge", "lasso", "elastic-net"):
        options = linear_suite.model_options(name, 0. if name == "ols" else .01, scale)
        ref = linear_suite.reference(xs[0], ys[0], options)
        assert ref["verified"]
        assert ref["stationarity_norm"] <= 1e-5
        assert np.isfinite(ref["objective"])
    a = linear_suite.reference(xs[0], ys[0], linear_suite.model_options("ols", 0., scale))
    b = linear_suite.reference(xs[0], ys[0], linear_suite.model_options("ols", 0., scale),
                              sample_weight=np.ones(len(ys[0])) * 1e200)
    assert a["objective"] == pytest.approx(b["objective"], abs=1e-10)


def test_quick_linear_cli(tmp_path):
    pytest.importorskip("sklearn")
    script = Path(linear_suite.__file__)
    child = subprocess.run([sys.executable, str(script), "--quick", "--cases", "positive_prior",
                            "--seeds", "42", "--max-iter", "2", "--repeats", "1",
                            "--output", str(tmp_path)], capture_output=True, text=True, check=True)
    data = json.loads((tmp_path / "results.json").read_text())
    assert {r["model"] for r in data["results"]} == {
        "ols", "ridge", "lasso", "elastic-net", "nonnegative", "box-fixed"}
    assert all(r["reference_valid"] for r in data["results"])
    assert all(r["test"]["mse"] >= 0 for r in data["results"])
    assert all(r["constraint_violation"] <= 1e-12 for r in data["results"])
    assert {r["solver"] for r in data["results"]} == {"cg", "proximal-gd", "fista"}
    assert (tmp_path / "traces.csv").is_file() and (tmp_path / "metrics.csv").is_file()
    assert "positive_prior" in child.stdout
    from benchmarks import linear_report
    linear_report.validate_complete(data)
    summaries = linear_report.summarize(data)
    for language in (False, True):
        text = linear_report.render(data, summaries, language, "../raw")
        assert "../raw/results.json" in text and "box-fixed" in text
    with pytest.raises(ValueError, match="Incomplete"):
        linear_report.validate_complete(data | {"results": data["results"][:-1]})
    base = data["results"][0]
    unequal_repeats = [base | dict(seed=42, repeat=i, fit_seconds=1.) for i in range(3)]
    unequal_repeats += [base | dict(seed=43, repeat=0, fit_seconds=100., constraint_violation=9e-9)]
    aggregate, = linear_report.summarize(data | {"results": unequal_repeats})
    assert aggregate["fit_seconds"]["median"] == 50.5
    assert aggregate["maximum_constraint_violation"] == 9e-9


def test_linear_report_validation():
    from benchmarks import linear_report
    with pytest.raises(ValueError, match="Incomplete"):
        linear_report.validate_complete(dict(config=dict(cases=["dense_control"], seeds=[42],
                                                        repeats=1), results=[], selections=[]))


def test_test_labels_do_not_select_penalties():
    xs, ys, weights, scale, truth, meta = linear_suite.prepare("dense_control", 42, "rms", True)
    a, trials_a = linear_suite.select_model("ridge", xs, ys, scale, None)
    altered = [part.copy() for part in ys]
    altered[2] = np.arange(len(ys[2])) * 1e9
    b, trials_b = linear_suite.select_model("ridge", xs, altered, scale, None)
    assert a[0] == b[0]
    assert [r["validation"] for r in trials_a] == [r["validation"] for r in trials_b]
