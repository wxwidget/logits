"""Numerical and validation contracts shared by the documented public API."""
import numpy as np
import pytest
from scipy import sparse
from logitlab import LogisticRegression, _core

SOLVERS = ["gd", "sgd", "adam", "lbfgs", "newton-cg"]

@pytest.fixture
def sample():
    rng = np.random.default_rng(19)
    X = rng.normal(size=(80, 4))
    y = (X[:, 0] - .6 * X[:, 1] > .2).astype(float)
    return X, y

@pytest.mark.parametrize("solver", SOLVERS)
def test_budget_and_history(solver, sample):
    X, y = sample
    model = LogisticRegression(solver=solver, max_epochs=1, tolerance=1e-14,
                               l2=.1, batch_size=17).fit(X, y)
    assert model.status_ == "max_epochs" and not model.converged_
    assert [h["epoch"] for h in model.history_] == [0, 1]
    assert model.history_[1]["seconds"] >= model.history_[0]["seconds"] >= 0
    options = model.options
    f, g = _core.evaluate(_core.Dataset.dense(X, y),
                          np.r_[model.coef_, model.intercept_], options)
    np.testing.assert_allclose(f, model.history_[-1]["loss"], atol=1e-13)
    np.testing.assert_allclose(np.linalg.norm(g), model.history_[-1]["gradient_norm"], atol=1e-13)

@pytest.mark.parametrize("solver", ["gd", "lbfgs", "newton-cg"])
def test_batch_line_search_is_monotonic(solver, sample):
    X, y = sample
    model = LogisticRegression(solver=solver, learning_rate=100., l2=.1).fit(X, y)
    assert np.all(np.diff([h["loss"] for h in model.history_]) <= 1e-12)

@pytest.mark.parametrize("solver", SOLVERS)
def test_initial_optimum(solver):
    model = LogisticRegression(solver=solver).fit(np.zeros((4, 3)), [0, 1, 0, 1])
    assert model.converged_ and model.status_ == "converged"
    assert len(model.history_) == 1
    np.testing.assert_array_equal(model.coef_, np.zeros(3))
    np.testing.assert_allclose(model.predict_proba(np.zeros((2, 3))), .5)

@pytest.mark.parametrize("solver", ["lbfgs", "newton-cg"])
def test_intercept_has_analytic_optimum(solver):
    X = np.zeros((4, 2))
    model = LogisticRegression(solver=solver, l2=100., tolerance=1e-9).fit(X, [0, 1, 1, 1])
    assert model.converged_
    np.testing.assert_allclose(model.intercept_, np.log(3), atol=1e-7)
    np.testing.assert_array_equal(model.coef_, [0, 0])
    np.testing.assert_allclose(model.predict_proba(X)[:, 1], .75, atol=1e-8)
    without = LogisticRegression(solver=solver, fit_intercept=False).fit(X, [0, 1, 1, 1])
    assert without.intercept_ == 0
    np.testing.assert_allclose(without.predict_proba(X), .5)

@pytest.mark.parametrize("solver", ["sgd", "adam"])
def test_seed_and_sparse_input_reproducibility(solver, sample):
    X, y = sample
    kw = dict(solver=solver, seed=73, batch_size=13, max_epochs=10)
    a = LogisticRegression(**kw).fit(X, y)
    b = LogisticRegression(**kw).fit(sparse.csr_matrix(X), y)
    np.testing.assert_allclose(a.coef_, b.coef_, atol=1e-12)
    np.testing.assert_allclose(a.intercept_, b.intercept_, atol=1e-12)

@pytest.mark.parametrize("variant", ["fortran", "float32", "reversed", "list"])
def test_input_layout_and_dtype(variant, sample):
    X, y = sample
    variants = dict(fortran=np.asfortranarray(X), float32=X.astype(np.float32),
                    reversed=X[:, ::-1], list=X.tolist())
    value = variants[variant]
    a = LogisticRegression(l2=.1).fit(value, y)
    b = LogisticRegression(l2=.1).fit(np.array(value, dtype=np.float64, order="C"), y)
    np.testing.assert_allclose(a.coef_, b.coef_, atol=1e-12)

@pytest.mark.parametrize("shape", [(0, 2), (2, 0), (4,), (2, 2, 1)])
def test_bad_feature_shape(shape):
    with pytest.raises(ValueError):
        LogisticRegression().fit(np.zeros(shape), np.zeros(shape[0]))

@pytest.mark.parametrize("name,value", [
    ("max_epochs", 0), ("batch_size", 0), ("history_size", 0),
    ("learning_rate", 0), ("learning_rate", -1), ("learning_rate", np.inf),
    ("learning_rate", np.nan), ("l2", -1), ("l2", np.inf), ("l2", np.nan),
    ("tolerance", 0), ("tolerance", -1), ("tolerance", np.nan),
])
def test_bad_options(name, value, sample):
    X, y = sample
    with pytest.raises(ValueError):
        LogisticRegression(**{name: value}).fit(X, y)

@pytest.mark.parametrize("value", [np.nan, np.inf, -.1, -1])
def test_bad_labels(value, sample):
    X, y = sample
    y[0] = value
    with pytest.raises(ValueError): LogisticRegression().fit(X, y)

@pytest.mark.parametrize("values,indices,offsets,rows,cols", [
    ([1., 2.], [1, 0], [0, 2], 1, 2),
    ([1., 2.], [0, 0], [0, 2], 1, 2),
    ([1.], [2], [0, 1], 1, 2),
    ([1.], [0], [1, 1], 1, 2),
    ([1.], [0], [0, 2], 1, 2),
    ([], [], [0, 1, 0], 2, 2),
    ([np.nan], [0], [0, 1], 1, 2),
])
def test_invalid_raw_csr(values, indices, offsets, rows, cols):
    with pytest.raises(ValueError):
        _core.Dataset.csr(values, indices, offsets, rows, cols, np.zeros(rows))

@pytest.mark.parametrize("kind", ["csr", "csc"])
def test_malformed_scipy_indices_are_rejected(kind):
    constructor = getattr(sparse, kind + "_matrix")
    X = constructor(([1.], [9], [0, 1, 1]), shape=(2, 2))
    with pytest.raises(ValueError): LogisticRegression().fit(X, [0, 1])

def test_sparse_duplicates_are_summed_without_mutating_input():
    X = sparse.csr_matrix(([1., 2., 3., -2.], [1, 0, 1, 0], [0, 3, 4]), shape=(2, 2))
    original = (X.data.copy(), X.indices.copy(), X.indptr.copy())
    a = LogisticRegression(l2=.1).fit(X, [1, 0])
    b = LogisticRegression(l2=.1).fit([[2., 4.], [-2., 0.]], [1, 0])
    np.testing.assert_allclose(a.coef_, b.coef_, atol=1e-12)
    for before, after in zip(original, (X.data, X.indices, X.indptr)):
        np.testing.assert_array_equal(before, after)

def test_regularization_convention_and_disabled_intercept(sample):
    X, y = sample
    d = _core.Dataset.dense(X, y)
    o = _core.Options()
    w = np.array([.2, -.3, .1, .4, 1.7])
    o.l2 = 0
    f0, g0 = _core.evaluate(d, w, o)
    o.l2 = .7
    f1, g1 = _core.evaluate(d, w, o)
    np.testing.assert_allclose(f1-f0, .35*(w[:-1] @ w[:-1]), atol=1e-14)
    np.testing.assert_allclose(np.array(g1)-g0, np.r_[.7*w[:-1], 0], atol=1e-14)
    o.fit_intercept = False
    f, g = _core.evaluate(d, w, o)
    w[-1] = -100
    np.testing.assert_allclose(_core.evaluate(d, w, o)[0], f)
    assert g[-1] == 0

@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_evaluator_rejects_nonfinite_parameters(value, sample):
    X, y = sample
    w = np.zeros(5); w[0] = value
    with pytest.raises(ValueError): _core.evaluate(_core.Dataset.dense(X, y), w, _core.Options())

def test_extreme_wrong_margins_and_prediction_input(sample):
    o = _core.Options(); o.l2 = 0
    f, g = _core.evaluate(_core.Dataset.dense([[1000.], [-1000.]], [0., 1.]), [1., 0.], o)
    assert f == 1000. and np.isfinite(g).all()
    X, y = sample
    model = LogisticRegression().fit(X, y)
    X[0, 0] = np.nan
    with pytest.raises(ValueError): model.predict_proba(X)

def test_failed_refit_preserves_previous_model(sample):
    X, y = sample
    model = LogisticRegression().fit(X, y)
    expected = model.predict_proba(X)
    with pytest.raises(ValueError): model.fit(X, y[:-1])
    np.testing.assert_array_equal(model.predict_proba(X), expected)

@pytest.mark.parametrize("l2", [-1., np.nan, np.inf])
def test_evaluator_rejects_invalid_l2(l2, sample):
    X, y = sample
    options = _core.Options(); options.l2 = l2
    with pytest.raises(ValueError):
        _core.evaluate(_core.Dataset.dense(X, y), np.zeros(5), options)

def test_evaluator_parameter_shape(sample):
    X, y = sample
    with pytest.raises(ValueError):
        _core.evaluate(_core.Dataset.dense(X, y), np.zeros(4), _core.Options())

def test_exhausted_line_search_is_distinct(sample):
    X, y = sample
    model = LogisticRegression(solver="gd", l2=.1, learning_rate=1e308, max_epochs=1).fit(X, y)
    assert model.status_ == "line_search_failed" and not model.converged_
    assert len(model.history_) == 1
    np.testing.assert_array_equal(model.coef_, np.zeros(4))

def test_nonfinite_stochastic_progress_raises(sample):
    X, y = sample
    with pytest.raises(RuntimeError, match="non-finite objective"):
        LogisticRegression(solver="sgd", l2=.1, learning_rate=1e308,
                           batch_size=1, max_epochs=1).fit(X, y)
