import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import minimize
from logitlab import LogisticRegression, _core

@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 5))
    y = (X[:, 0] + .4 * X[:, 1] > 0).astype(float)
    return X, y

@pytest.mark.parametrize('solver', ['gd', 'sgd', 'adam', 'lbfgs', 'newton-cg'])
def test_dense_sparse(solver, data):
    X, y = data
    kw = dict(solver=solver, max_epochs=150, l2=.1, learning_rate=.1, batch_size=32)
    a = LogisticRegression(**kw).fit(X, y)
    b = LogisticRegression(**kw).fit(sparse.csr_matrix(X), y)
    np.testing.assert_allclose(a.coef_, b.coef_, atol=1e-10)
    assert a.history_[-1]['loss'] < a.history_[0]['loss']
    assert np.mean(a.predict(X) == y) > .9
    assert np.isfinite(a.predict_proba(X)).all()
    np.testing.assert_allclose(a.predict_proba(X).sum(axis=1), 1)

@pytest.mark.parametrize('solver', ['lbfgs', 'newton-cg'])
def test_against_scipy(solver, data):
    X, y = data
    def f(w):
        z = X @ w[:-1] + w[-1]
        loss = np.mean(np.logaddexp(0, z) - y * z) + .05 * (w[:-1] @ w[:-1])
        p = 1 / (1 + np.exp(-z))
        g = np.r_[X.T @ (p - y) / len(y) + .1 * w[:-1], np.mean(p - y)]
        return loss, g
    reference = minimize(f, np.zeros(X.shape[1] + 1), jac=True, method='BFGS', options={'gtol': 1e-9})
    model = LogisticRegression(solver=solver, l2=.1, tolerance=1e-8, max_epochs=100).fit(X,y)
    assert model.converged_
    np.testing.assert_allclose(np.r_[model.coef_, model.intercept_], reference.x, atol=1e-6)
    assert abs(model.history_[-1]['loss'] - reference.fun) < 1e-10

@pytest.mark.parametrize('bad', [np.nan, np.inf])
def test_bad_features(bad, data):
    X,y=data;X[0,0]=bad
    with pytest.raises(ValueError): LogisticRegression().fit(X,y)

def test_invalid_inputs(data):
    X,y=data
    for bad_y in [y[:-1], y[:,None], np.full_like(y,2)]:
        with pytest.raises(ValueError): LogisticRegression().fit(X,bad_y)
    with pytest.raises(ValueError): LogisticRegression(solver='invalid').fit(X,y)
    with pytest.raises(RuntimeError): LogisticRegression().predict(X)
    m=LogisticRegression().fit(X,y)
    with pytest.raises(ValueError): m.predict(X[:,:-1])
    with pytest.raises(ValueError): _core.Dataset.csr([1.], [-1], [0,1], 1,1,y[:1])

def test_canonicalization(data):
    X,y=data
    a=sparse.coo_matrix(X)
    m=LogisticRegression().fit(a,y)
    assert np.mean(m.predict(a)==y)>.9

def test_gradient_and_extreme_margins(data):
    X,y=data;d=_core.Dataset.dense(X,y);o=_core.Options();w=np.linspace(-.3,.3,6)
    _,g=_core.evaluate(d,w,o)
    numerical=[]
    for j in range(6):
        delta=np.zeros(6);delta[j]=1e-6
        numerical.append((_core.evaluate(d,w+delta,o)[0]-_core.evaluate(d,w-delta,o)[0])/2e-6)
    np.testing.assert_allclose(g,numerical,atol=1e-8)
    extreme=_core.Dataset.dense([[1000.],[-1000.]], [1.,0.])
    f,g=_core.evaluate(extreme,[1.,0.],o)
    assert np.isfinite(f) and np.isfinite(g).all()

def test_seed_and_intercept(data):
    X,y=data
    a=LogisticRegression(solver='sgd',seed=7).fit(X,y)
    b=LogisticRegression(solver='sgd',seed=7).fit(X,y)
    np.testing.assert_array_equal(a.coef_,b.coef_)
    assert LogisticRegression(fit_intercept=False).fit(X,y).intercept_==0
