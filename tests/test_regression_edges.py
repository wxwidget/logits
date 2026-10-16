"""Numerical edge regressions added in a second RED/GREEN cycle."""
import numpy as np
import pytest
from scipy import stats
from logitlab import LinearRegression, regression_metrics


@pytest.mark.parametrize("solver", ["cg", "proximal-gd", "fista"])
def test_zero_weight_rows_are_truly_excluded(solver):
    X, y = np.array([[-1.], [1.], [1e308]]), np.array([-2., 2., 0.])
    model = LinearRegression(solver=solver, tolerance=1e-9).fit(X, y, sample_weight=[1., 1., 0.])
    assert model.converged_
    np.testing.assert_allclose(model.predict(X[:2]), [-2., 2.], atol=1e-8)


def test_constant_target_r2_uses_finite_policy_with_roundoff():
    result = regression_metrics(np.full(7, 2.), np.ones(7))
    assert result["r2"] == 0
    assert regression_metrics(np.full(7, 2.), np.full(7, 2.))["r2"] == 1


def test_large_intercept_is_not_mistaken_for_zero_residual_variance():
    rng = np.random.default_rng(31)
    X = rng.normal(size=(130, 1))
    y = 1e8 + 2 * X[:, 0] + rng.normal(scale=.8, size=len(X))
    model = LinearRegression().fit(X, y)
    assert model.converged_
    summary = model.summary(X, y)
    reference = stats.linregress(X[:, 0], y)
    assert summary["coefficients"][0]["standard_error"] == pytest.approx(reference.stderr, rel=1e-6)


def test_no_intercept_summary_and_weight_rescaling():
    rng = np.random.default_rng(33)
    X = rng.normal(size=(140, 2)); y = X @ [1., -2.] + rng.normal(scale=.2, size=len(X))
    weights = np.linspace(.1, 2., len(y))
    model = LinearRegression(fit_intercept=False).fit(X, y, sample_weight=weights)
    a = model.summary(X, y, sample_weight=weights)
    b = model.summary(X, y, sample_weight=weights * 1e200)
    assert a["df_model"] == 2 and a["df_resid"] == 138
    np.testing.assert_allclose(a["covariance"], b["covariance"], atol=1e-12)
    assert a["aic"] == pytest.approx(b["aic"])
    e = y - model.predict(X)
    expected_r2 = 1 - np.sum(weights * e**2) / np.sum(weights * y**2)
    assert a["r2"] == pytest.approx(expected_r2)


def test_overflowed_regression_prediction_is_rejected():
    model = LinearRegression().fit([[-1.], [1.]], [-2., 2.])
    with pytest.raises(RuntimeError, match="non-finite"):
        model.predict([[1e308]])


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("target", [2., 1e308])
def test_constant_fit_has_exact_intercept_and_perfect_finite_score(weighted, target):
    X = np.arange(7, dtype=float)[:, None]
    y = np.full(7, target)
    weights = np.arange(7, dtype=float) if weighted else None
    if weighted: y[0] = -target  # This observation must not influence initialization.
    model = LinearRegression().fit(X, y, sample_weight=weights)
    assert model.converged_ and model.intercept_ == target
    assert model.score(X, y, sample_weight=weights) == 1.
