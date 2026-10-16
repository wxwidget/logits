"""OLS inference checked against independent SciPy/formula references."""
import numpy as np
import pytest
from scipy import sparse, stats
from logitlab import LinearRegression, Ridge, regression_metrics


def sample():
    rng = np.random.default_rng(201)
    X = rng.normal(size=(130, 3))
    y = X @ [2., -.5, 1.] + 3 + rng.normal(scale=.8, size=len(X))
    return X, y


def test_simple_regression_matches_scipy():
    X, y = sample(); X = X[:, :1]
    reference = stats.linregress(X[:, 0], y)
    m = LinearRegression(tolerance=1e-10).fit(X, y)
    summary = m.summary(X, y)
    slope, intercept = summary["coefficients"]
    assert slope["estimate"] == pytest.approx(reference.slope, abs=1e-9)
    assert slope["standard_error"] == pytest.approx(reference.stderr)
    assert intercept["standard_error"] == pytest.approx(reference.intercept_stderr)
    assert slope["p_value"] == pytest.approx(reference.pvalue)
    assert summary["r2"] == pytest.approx(reference.rvalue**2)
    assert slope["ci_lower"] < slope["estimate"] < slope["ci_upper"]
    assert summary["df_resid"] == len(y) - 2


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("covariance", ["classical", "HC3"])
def test_multivariate_covariance_and_influence(weighted, covariance):
    X, y = sample(); n = len(y)
    weights = np.geomspace(.1, 2, n) if weighted else np.ones(n)
    weights[0] = 0 if weighted else 1
    m = LinearRegression(tolerance=1e-10).fit(X, y, sample_weight=weights)
    s = m.summary(X, y, sample_weight=weights, covariance=covariance)
    mask = weights > 0; A = np.c_[X[mask], np.ones(mask.sum())]
    w = weights[mask] / weights.max(); e = y[mask] - m.predict(X[mask])
    inv = np.linalg.inv(A.T @ (w[:, None] * A))
    h = np.einsum("ij,jk,ik->i", A, inv, A) * w
    sigma2 = np.sum(w * e**2) / (mask.sum() - 4)
    if covariance == "classical":
        expected = sigma2 * inv
    else:
        meat = A.T @ (((w * e / (1 - h))**2)[:, None] * A)
        expected = inv @ meat @ inv
    np.testing.assert_allclose(s["covariance"], expected, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(s["leverage"], h, atol=1e-10)
    assert s["rank"] == 4 and s["nobs"] == mask.sum()
    assert s["condition_number"] >= 1 and np.isfinite(s["aic"]) and np.isfinite(s["bic"])
    assert np.all(s["cooks_distance"] >= 0)
    assert s["adjusted_r2"] < s["r2"]


def test_inference_is_restricted_to_valid_ols():
    X, y = sample()
    for m in (Ridge(alpha=.1), LinearRegression(positive=True),
              LinearRegression(fixed_coefficients={1: 0.}),
              LinearRegression(solver="proximal-gd", max_iter=1, tolerance=1e-14)):
        m.fit(X, y)
        with pytest.raises(ValueError): m.summary(X, y)
    m = LinearRegression().fit(X, y)
    with pytest.raises(ValueError): m.summary(X, y[::-1])
    with pytest.raises(ValueError): m.summary(X, y, confidence=1.)
    with pytest.raises(ValueError): m.summary(X, y, covariance="invented")
    with pytest.raises(ValueError): m.summary(X, y, max_features=1)
    with pytest.raises(ValueError): m.summary(sparse.csr_matrix(X), y)
    duplicate = np.column_stack((X, X[:, 0]))
    with pytest.raises(ValueError): LinearRegression().fit(duplicate, y).summary(duplicate, y)
    tiny = X[:4]
    with pytest.raises(ValueError): LinearRegression().fit(tiny, y[:4]).summary(tiny, y[:4])


def test_metrics_validation_and_nonfinite_predictions():
    for y, p in [([], []), ([1., 2.], [1.]), ([[1.]], [[1.]]), ([np.nan], [0.]), ([1.], [np.inf])]:
        with pytest.raises(ValueError): regression_metrics(y, p)
    X, y = sample()
    m = LinearRegression().fit(X, y)
    with pytest.raises(ValueError): m.residuals(X, y[:-1])


def test_inference_resource_and_degeneracy_guards():
    X, y = sample()
    with pytest.raises(RuntimeError): LinearRegression().summary(X, y)
    m = LinearRegression().fit(X, y)
    with pytest.raises(ValueError): m.summary(X[:-1], y)
    large = np.broadcast_to(np.zeros((1, 3)), (1_300_000, 3))
    with pytest.raises(ValueError, match="5 million"):
        m.summary(large, np.zeros(len(large)))
    zero = LinearRegression().fit(X, np.zeros(len(X)))
    with pytest.raises(ValueError, match="residual variance"): zero.summary(X, np.zeros(len(X)))
    # A coefficient dedicated to a single observation gives unit leverage.
    design = np.column_stack((np.arange(12.), np.r_[1., np.zeros(11)]))
    response = np.random.default_rng(11).normal(size=12)
    unit = LinearRegression().fit(design, response)
    with pytest.raises(ValueError, match="unit leverage"): unit.summary(design, response, covariance="HC3")
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="overflow"):
            regression_metrics([1e308, -1e308], [-1e308, 1e308])
