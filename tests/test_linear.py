"""Independent solution contracts, written before the linear implementation."""
import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import lsq_linear, minimize

from logitlab import ElasticNet, Lasso, LinearRegression, Ridge, regression_metrics


@pytest.fixture
def data():
    rng = np.random.default_rng(190)
    X = rng.normal(size=(180, 6))
    y = X @ np.array([2., -1., 0., .5, 0., -2.]) + 3 + rng.normal(scale=.2, size=180)
    return X, y


@pytest.mark.parametrize("solver", ["cg", "proximal-gd", "fista"])
@pytest.mark.parametrize("l2", [0., .2])
@pytest.mark.parametrize("csr", [False, True])
def test_least_squares_and_ridge_reference(data, solver, l2, csr):
    X, y = data
    A = np.column_stack((X, np.ones(len(y))))
    reg = np.diag(np.r_[np.full(X.shape[1], np.sqrt(l2)), 0.])
    reference = np.linalg.lstsq(np.vstack((A / np.sqrt(len(y)), reg)),
                                np.r_[y / np.sqrt(len(y)), np.zeros(A.shape[1])], rcond=None)[0]
    model = LinearRegression(solver=solver, l2=l2, tolerance=1e-9, max_iter=2000)
    model.fit(sparse.csr_matrix(X) if csr else X, y)
    assert model.converged_ and model.status_ == "converged"
    np.testing.assert_allclose(np.r_[model.coef_, model.intercept_], reference, atol=2e-7)
    np.testing.assert_allclose(model.predict(X), A @ reference, atol=8e-7)
    reference_r2 = 1 - np.sum((y - A @ reference)**2) / np.sum((y - y.mean())**2)
    assert model.score(X, y) == pytest.approx(reference_r2, abs=1e-9)
    assert model.history_[-1]["stationarity_norm"] <= 1e-9
    assert model.work_["full_evaluations"] >= len(model.history_)


@pytest.mark.parametrize("solver", ["auto", "cg", "proximal-gd", "fista"])
def test_weighted_fit_and_weight_scaling(data, solver):
    X, y = data
    weights = np.geomspace(.01, 10, len(y)); weights[::7] = 0
    q = weights / weights.sum()
    A = np.column_stack((X, np.ones(len(y))))
    expected = np.linalg.lstsq(A * np.sqrt(q[:, None]), y * np.sqrt(q), rcond=None)[0]
    a = LinearRegression(solver=solver, tolerance=1e-9, max_iter=2000).fit(X, y, sample_weight=weights)
    b = LinearRegression(solver=solver, tolerance=1e-9, max_iter=2000).fit(
        sparse.csr_matrix(X), y, sample_weight=weights * 1e200)
    np.testing.assert_allclose(np.r_[a.coef_, a.intercept_], expected, atol=3e-7)
    np.testing.assert_allclose(a.predict(X), b.predict(X), atol=1e-7)


@pytest.mark.parametrize("solver", ["proximal-gd", "fista"])
@pytest.mark.parametrize("l2", [0., .3])
def test_l1_orthogonal_closed_form(solver, l2):
    # Balanced orthogonal columns, X.T X / n = I, intercept orthogonal.
    X = np.array([[1., 1., 1.], [1., -1., -1.], [-1., 1., -1.], [-1., -1., 1.]])
    truth = np.array([2., -.4, .05]); l1 = .2
    y = X @ truth + 4
    expected = np.sign(truth) * np.maximum(abs(truth) - l1, 0) / (1 + l2)
    for features in (X, sparse.csr_matrix(X)):
        model = LinearRegression(solver=solver, l1=l1, l2=l2, tolerance=1e-10).fit(features, y)
        np.testing.assert_allclose(model.coef_, expected, atol=1e-9)
        assert model.intercept_ == pytest.approx(4.)
        assert model.coef_[2] == 0  # Actual sparsity, not a tolerance threshold.
        assert model.solver_ == solver and model.converged_


@pytest.mark.parametrize("solver", ["proximal-gd", "fista"])
def test_box_and_fixed_coefficients_against_scipy(data, solver):
    X, y = data
    lower = np.array([0., -2., .25, -.1, -np.inf, -3.])
    upper = np.array([.5, 0., .25, .1, np.inf, -1.])
    # Eliminate the fixed coordinate for SciPy's strictly separated bounds.
    free = np.array([0, 1, 3, 4, 5])
    A = np.column_stack((X[:, free], np.ones(len(y))))
    ref = lsq_linear(A, y - .25 * X[:, 2], bounds=(np.r_[lower[free], 1.],
                                                                np.r_[upper[free], 2.]))
    model = LinearRegression(solver=solver, bounds=(lower, upper), intercept_bounds=(1., 2.),
                             tolerance=1e-9, max_iter=3000).fit(X, y)
    assert model.converged_
    assert np.all(model.coef_ >= lower) and np.all(model.coef_ <= upper)
    assert model.coef_[2] == .25 and 1 <= model.intercept_ <= 2
    np.testing.assert_allclose(np.r_[model.coef_[free], model.intercept_], ref.x, atol=1e-6)
    # Fixed-value convenience is equivalent, including nonzero initialization.
    b = LinearRegression(solver=solver, bounds=(lower, upper),
                         fixed_coefficients={2: .25}, intercept_bounds=(1., 2.),
                         tolerance=1e-9, max_iter=3000).fit(sparse.csr_matrix(X), y)
    np.testing.assert_allclose(model.predict(X), b.predict(X), atol=1e-7)
    losses = [h["loss"] for h in model.history_]
    assert np.all(np.diff(losses) <= 1e-12)


def test_positive_elastic_net_kkt(data):
    X, y = data
    l1, l2 = .1, .2
    ref = minimize(lambda w: (.5 * np.mean((X @ w[:-1] + w[-1] - y)**2) +
                               l1 * w[:-1].sum() + .5 * l2 * (w[:-1] @ w[:-1])),
                   np.zeros(7), method="L-BFGS-B", bounds=[(0, None)] * 6 + [(None, None)],
                   options={"ftol": 1e-14, "gtol": 1e-9})
    model = LinearRegression(l1=l1, l2=l2, positive=True, tolerance=1e-9).fit(X, y)
    assert model.solver_ == "fista" and model.converged_
    assert np.all(model.coef_ >= 0)
    np.testing.assert_allclose(model.predict(X), X @ ref.x[:-1] + ref.x[-1], atol=1e-5)


def test_model_aliases(data):
    X, y = data
    for alias, direct in [
        (Ridge(alpha=.3), LinearRegression(l2=.3)),
        (Lasso(alpha=.1), LinearRegression(l1=.1)),
        (ElasticNet(alpha=.2, l1_ratio=.25), LinearRegression(l1=.05, l2=.15)),
    ]:
        np.testing.assert_allclose(alias.fit(X, y).predict(X), direct.fit(X, y).predict(X), atol=1e-9)


def test_named_models_reject_conflicting_regularizers():
    with pytest.raises(TypeError): Ridge(alpha=.1, l1=.2)
    with pytest.raises(TypeError): Lasso(alpha=.1, l2=.2)
    with pytest.raises(ValueError): ElasticNet(l1_ratio=1.1)


def test_rank_deficient_zero_columns_and_disabled_intercept():
    X = np.array([[1., 1., 0.], [2., 2., 0.], [3., 3., 0.], [4., 4., 0.]])
    y = np.array([1., 2., 2., 4.])
    for solver in ("cg", "proximal-gd", "fista"):
        m = LinearRegression(solver=solver, fit_intercept=False, tolerance=1e-9).fit(X, y)
        ref = X @ np.linalg.lstsq(X, y, rcond=None)[0]
        np.testing.assert_allclose(m.predict(X), ref, atol=1e-7)
        assert m.intercept_ == 0
        np.testing.assert_array_equal(m.predict(X), m.predict(sparse.csr_matrix(X)))


def test_fixed_model_is_stationary_with_nonzero_gradient():
    m = LinearRegression(fixed_coefficients={0: 2.}, intercept_bounds=(3., 3.)).fit(
        [[-1.], [1.]], [-100., 100.])
    assert m.converged_ and m.history_[-1]["stationarity_norm"] == 0
    assert m.coef_[0] == 2. and m.intercept_ == 3.


def test_invalid_inputs_and_failed_refit(data):
    X, y = data
    m = LinearRegression().fit(X, y); before = m.predict(X)
    for bad in (y[:-1], y[:, None], np.full_like(y, np.nan), np.full_like(y, np.inf)):
        with pytest.raises(ValueError): m.fit(X, bad)
        np.testing.assert_array_equal(m.predict(X), before)
    for weights in ([0.] * len(y), [-1.] * len(y), [1.], np.full(len(y), np.inf),
                    np.ones((len(y), 1)), np.full(len(y), np.nan)):
        with pytest.raises(ValueError): m.fit(X, y, sample_weight=weights)
    with pytest.raises(RuntimeError): LinearRegression().predict(X)
    with pytest.raises(ValueError): m.predict(X[:, :-1])
    with pytest.raises(ValueError): m.predict(np.full_like(X, np.inf))


@pytest.mark.parametrize("options", [
    {"l1": -1}, {"l2": np.nan}, {"l2": np.inf}, {"max_iter": 0}, {"tolerance": 0},
    {"learning_rate": 0}, {"solver": "bad"}, {"solver": "cg", "l1": .1},
    {"solver": "cg", "positive": True}, {"bounds": (1., 0.)},
    {"bounds": (np.nan, 1.)}, {"bounds": (np.inf, np.inf)}, {"bounds": (-np.inf, -np.inf)},
    {"bounds": ([0., 1.], None)}, {"fixed_coefficients": {7: 0.}},
    {"fixed_coefficients": {-1: 0.}}, {"fixed_coefficients": {1.5: 0.}},
    {"fixed_coefficients": {0: np.nan}}, {"positive": True, "fixed_coefficients": {0: -1.}},
    {"bounds": (0., 1.), "fixed_coefficients": {0: 2.}},
    {"intercept_bounds": (2., 1.)}, {"intercept_bounds": (np.nan, 1.)},
    {"fit_intercept": False, "intercept_bounds": (1., 2.)},
])
def test_invalid_options(data, options):
    with pytest.raises((ValueError, TypeError)):
        LinearRegression(**options).fit(*data)


def test_metrics_and_budget(data):
    m = regression_metrics([1., 2., 3.], [2., 2., 2.])
    assert m["mse"] == pytest.approx(2 / 3)
    assert m["rmse"] == pytest.approx(np.sqrt(2 / 3))
    assert m["mae"] == pytest.approx(2 / 3) and m["r2"] == 0
    assert regression_metrics([2., 2.], [2., 2.])["r2"] == 1
    assert regression_metrics([2., 2.], [1., 1.])["r2"] == 0
    assert regression_metrics([0., 10.], [1., 0.], sample_weight=[1., 0.])["mse"] == 1
    X, y = data
    model = LinearRegression(solver="proximal-gd", max_iter=1, tolerance=1e-14).fit(X, y)
    assert not model.converged_ and model.status_ == "max_iter"
    assert model.evaluate(X, y) == regression_metrics(y, model.predict(X))
    np.testing.assert_allclose(model.residuals(X, y), y - model.predict(X))


def test_prediction_inspection_copy_and_mutated_options(data):
    X, y = data
    m = LinearRegression().fit(X, y); expected = m.predict(X)
    m.coef_[:] = 999.; m.intercept_ = 999.
    m.options.l1 = 100.
    np.testing.assert_array_equal(m.predict(X), expected)


@pytest.mark.parametrize("csr", [False, True])
def test_weighted_elastic_net_against_independent_coordinate_descent(data, csr):
    sklearn = pytest.importorskip("sklearn.linear_model")
    X, y = data
    weights = np.geomspace(.01, 10., len(y)); weights[::9] = 0
    features = sparse.csr_matrix(X) if csr else X
    reference = sklearn.ElasticNet(alpha=.05, l1_ratio=.6, tol=1e-11, max_iter=10000).fit(
        features, y, sample_weight=weights)
    model = ElasticNet(alpha=.05, l1_ratio=.6, tolerance=1e-8).fit(features, y, sample_weight=weights)
    assert model.converged_
    np.testing.assert_allclose(model.coef_, reference.coef_, atol=1e-6)
    assert model.intercept_ == pytest.approx(reference.intercept_, abs=1e-6)
