"""Canonical GLM contracts checked against independent formulations."""
import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit, xlogy

from logitlab import GeneralizedLinearModel, LinearRegression, LogisticRegression, _core


def problem(family, sparse_input=False):
    rng = np.random.default_rng(182)
    X = rng.normal(size=(240, 5))
    X[rng.random(X.shape) < .5] = 0
    offset = rng.normal(scale=.2, size=len(X))
    eta = X @ [.4, -.3, .2, 0, .1] + .3 + offset
    y = (rng.normal(eta, .7) if family == "gaussian" else
         rng.binomial(1, expit(eta)).astype(float) if family == "binomial" else rng.poisson(np.exp(eta)).astype(float))
    return sparse.csr_matrix(X) if sparse_input else X, y, offset


def reference(X, y, family, weights, offset, l2, fit_intercept):
    A = X.toarray() if sparse.issparse(X) else X
    A = np.column_stack((A, np.ones(len(y)))) if fit_intercept else A
    q = weights / weights.sum()
    def fun(theta):
        eta = A @ theta + offset
        if family == "gaussian": loss, residual = .5*(eta-y)**2, eta-y
        elif family == "binomial": loss, residual = np.logaddexp(0, eta)-y*eta, expit(eta)-y
        else: loss, residual = np.exp(eta)-y*eta, np.exp(eta)-y
        penalty = theta.copy()
        if fit_intercept: penalty[-1] = 0
        return q @ loss + .5*l2*(penalty@penalty), A.T@(q*residual)+l2*penalty
    result = minimize(fun, np.zeros(A.shape[1]), jac=True, method="BFGS", options=dict(gtol=1e-10,maxiter=1000))
    assert np.linalg.norm(fun(result.x)[1]) < 1e-7
    return result.x


@pytest.mark.parametrize("family", ["gaussian", "binomial", "poisson"])
@pytest.mark.parametrize("sparse_input", [False, True])
@pytest.mark.parametrize("fit_intercept", [False, True])
def test_independent_solutions_and_weight_scaling(family, sparse_input, fit_intercept):
    X, y, offset = problem(family, sparse_input)
    weights = np.linspace(0, 2, len(y))
    model = GeneralizedLinearModel(family=family, l2=.03, fit_intercept=fit_intercept,
                                   tolerance=1e-9).fit(X, y, sample_weight=weights, offset=offset)
    expected = reference(X, y, family, weights, offset, .03, fit_intercept)
    actual = np.r_[model.coef_, model.intercept_] if fit_intercept else model.coef_
    np.testing.assert_allclose(actual, expected, atol=2e-7)
    assert model.converged_ and model.status_ == "converged"
    assert model.family_ == family
    assert model.link_ == {"gaussian":"identity","binomial":"logit","poisson":"log"}[family]
    assert model.history_[-1]["gradient_norm"] <= 1e-9
    assert model.work_["hessian_vector_products"] > 0
    assert model.work_["full_evaluations"] >= len(model.history_)
    assert all(b["loss"] <= a["loss"]+1e-12 for a,b in zip(model.history_,model.history_[1:]))
    other = GeneralizedLinearModel(family=family, l2=.03, fit_intercept=fit_intercept,
                                   tolerance=1e-9).fit(X,y,sample_weight=weights*1e200,offset=offset)
    np.testing.assert_allclose(model.coef_,other.coef_,atol=1e-8)


@pytest.mark.parametrize("family", ["gaussian","binomial","poisson"])
def test_deviance_link_and_mean_predictions(family):
    X,y,offset = problem(family)
    weights = np.arange(1,len(y)+1,dtype=float)
    model = GeneralizedLinearModel(family=family).fit(X,y,offset=offset)
    eta = model.decision_function(X,offset=offset)
    np.testing.assert_allclose(eta,X@model.coef_+model.intercept_+offset)
    mu = model.predict(X,offset=offset)
    expected = eta if family == "gaussian" else expit(eta) if family == "binomial" else np.exp(eta)
    np.testing.assert_allclose(mu,expected)
    if family == "gaussian": unit = (y-mu)**2
    elif family == "binomial": unit = 2*(xlogy(y,y/mu)+xlogy(1-y,(1-y)/(1-mu)))
    else: unit = 2*(xlogy(y,y/mu)-y+mu)
    assert model.deviance(X,y,sample_weight=weights,offset=offset) == pytest.approx(np.average(unit,weights=weights))
    if family == "binomial":
        np.testing.assert_allclose(model.predict_proba(X,offset=offset)[:,1],mu)
    else:
        with pytest.raises(ValueError,match="binomial"): model.predict_proba(X)


def test_matches_existing_gaussian_and_logistic_models():
    for family in ("gaussian","binomial"):
        X,y,_ = problem(family)
        glm = GeneralizedLinearModel(family=family,l2=.02,tolerance=1e-9).fit(X,y)
        old = (LinearRegression(l2=.02,tolerance=1e-9) if family == "gaussian" else
               LogisticRegression(l2=.02,tolerance=1e-9,max_epochs=1000)).fit(X,y)
        np.testing.assert_allclose(glm.coef_,old.coef_,atol=1e-6)


def test_exposure_is_log_offset_and_must_be_resupplied():
    X,y,offset = problem("poisson",True)
    exposure = np.exp(offset)
    a = GeneralizedLinearModel(family="poisson",tolerance=1e-9).fit(X,y,exposure=exposure)
    b = GeneralizedLinearModel(family="poisson",tolerance=1e-9).fit(X,y,offset=offset)
    np.testing.assert_allclose(a.coef_,b.coef_,atol=1e-9)
    np.testing.assert_allclose(a.predict(X,exposure=exposure),b.predict(X,offset=offset))
    combined = a.decision_function(X,offset=np.ones(len(y)),exposure=exposure)
    np.testing.assert_allclose(combined,a.decision_function(X,exposure=exposure)+1)
    assert not np.allclose(a.predict(X),a.predict(X,exposure=exposure))


def test_binomial_proportions_and_trials_weight():
    X,y,_ = problem("binomial")
    y = .1 + .8*y
    weights = np.linspace(1,10,len(y))
    glm = GeneralizedLinearModel(family="binomial",tolerance=1e-9).fit(X,y,sample_weight=weights)
    expected = reference(X,y,"binomial",weights,np.zeros(len(y)),0,True)
    np.testing.assert_allclose(np.r_[glm.coef_,glm.intercept_],expected,atol=1e-7)


@pytest.mark.parametrize("options", [dict(family="gamma"),dict(family="poisson",link="identity"),
    dict(family="gaussian",link="log"),dict(l2=-1),dict(l2=np.nan),dict(tolerance=0),dict(max_iter=0),dict(max_cg_iter=0)])
def test_invalid_options(options):
    with pytest.raises(ValueError): GeneralizedLinearModel(**options).fit([[0.],[1.]],[0.,1.])


@pytest.mark.parametrize("family,y", [("binomial",[-.1,1]),("binomial",[0,1.1]),("poisson",[-1,2])])
def test_response_domain(family,y):
    with pytest.raises(ValueError): GeneralizedLinearModel(family=family).fit([[0.],[1.]],y)


@pytest.mark.parametrize("argument,value", [("offset",[1]),("offset",[0,np.nan]),
    ("exposure",[1,0]),("exposure",[1,-1]),("exposure",[1,np.inf]),
    ("sample_weight",[0,0]),("sample_weight",[-1,2]),("sample_weight",[1])])
def test_invalid_row_metadata(argument,value):
    with pytest.raises(ValueError): GeneralizedLinearModel(family="poisson").fit([[0.],[1.]],[0.,1.],**{argument:value})


def test_exposure_family_guard_and_fitted_state():
    model = GeneralizedLinearModel()
    with pytest.raises(RuntimeError): model.predict([[0.]])
    with pytest.raises(ValueError): model.fit([[0.],[1.]],[0.,1.],exposure=[1,1])
    X,y,offset = problem("poisson")
    model = GeneralizedLinearModel(family="poisson").fit(X,y,offset=offset)
    prediction = model.predict(X,offset=offset)
    with pytest.raises(ValueError): model.fit(X,-np.ones(len(y)))
    model.options.family = "gaussian"
    model.coef_[:] = 1000; model.intercept_ = 1000
    np.testing.assert_array_equal(model.predict(X,offset=offset),prediction)
    with pytest.raises(ValueError): model.predict(X[:,:-1])
    with pytest.raises(ValueError): model.predict(X,exposure=np.zeros(len(y)))


def test_zero_weight_rows_do_not_overflow_and_budget_status():
    X = np.r_[np.arange(10.)[:,None]/10, [[1e308]]]
    y = np.r_[np.arange(10.)+1, 1]
    offset = np.r_[np.zeros(10),1000.]
    model = GeneralizedLinearModel(family="poisson",tolerance=1e-9).fit(X,y,sample_weight=np.r_[np.ones(10),0],offset=offset)
    reference_model = GeneralizedLinearModel(family="poisson",tolerance=1e-9).fit(X[:10],y[:10])
    np.testing.assert_allclose(model.coef_,reference_model.coef_,atol=1e-8)
    X,y,_ = problem("poisson")
    limited = GeneralizedLinearModel(family="poisson",max_iter=1,tolerance=1e-14).fit(X,y)
    assert not limited.converged_ and limited.status_ == "max_iter"
    assert np.isfinite(limited.predict(X)).all()


def test_stable_large_offset_and_prediction_overflow():
    X = np.zeros((10,1)); y = np.arange(1.,11.)
    model = GeneralizedLinearModel(family="poisson",tolerance=1e-10).fit(X,y,offset=np.full(10,1000.))
    np.testing.assert_allclose(model.predict(X,offset=np.full(10,1000.)),np.full(10,5.5),rtol=1e-10)
    with pytest.raises(RuntimeError): model.predict(X,offset=np.full(10,2000.))


def test_core_gradient_with_disabled_intercept():
    X,y,offset = problem("poisson")
    data = _core.Dataset.dense(X,y,False)
    options = _core.GLMOptions(); options.family="poisson"; options.fit_intercept=False
    theta = np.r_[np.zeros(X.shape[1]),0.]
    loss,gradient = _core.evaluate_glm(data,theta,options,[],offset)
    assert np.isfinite(loss) and gradient[-1] == 0
    theta[-1] = 1
    with pytest.raises(ValueError): _core.evaluate_glm(data,theta,options,[],offset)


def test_poisson_backtracks_overflow_instead_of_clipping_eta():
    X = np.ones((20,1)); y = np.full(20,1000.)
    model = GeneralizedLinearModel(family="poisson",fit_intercept=False,tolerance=1e-8).fit(X,y)
    assert model.converged_
    np.testing.assert_allclose(model.coef_,[np.log(1000)],atol=1e-9)
    assert model.work_["full_evaluations"] > len(model.history_)


def test_saturated_binomial_losses_remain_finite():
    data = _core.Dataset.dense([[-1.],[1.]],[0.,1.],False)
    options = _core.GLMOptions(); options.family="binomial"
    loss,gradient = _core.evaluate_glm(data,[1000.,0.],options)
    assert loss == 0 and np.isfinite(gradient).all()
    loss,gradient = _core.evaluate_glm(data,[-1000.,0.],options)
    assert loss == 1000 and np.isfinite(gradient).all()


def test_deviance_uses_eta_when_probability_rounds_to_boundary():
    X = np.zeros((20,1)); y = np.full(20,.5)
    model = GeneralizedLinearModel(family="binomial").fit(X,y)
    assert model.deviance(X,y,offset=np.full(20,1000.)) == pytest.approx(1000.-2*np.log(2))
    poisson = GeneralizedLinearModel(family="poisson").fit(X,np.ones(20))
    assert poisson.deviance(X,np.ones(20),offset=np.full(20,-1000.)) == pytest.approx(1998.)


@pytest.mark.parametrize("target",[2.,1e308])
def test_gaussian_constant_exact_intercept(target):
    X = np.arange(7.)[:,None]; y = np.full(7,target)
    model = GeneralizedLinearModel().fit(X,y)
    assert model.intercept_ == target and model.converged_
    assert model.deviance(X,y) == 0


def test_poisson_agrees_with_sklearn():
    pytest.importorskip("sklearn")
    from sklearn.linear_model import PoissonRegressor
    X,y,_ = problem("poisson",True)
    weights = np.linspace(1,3,len(y))
    native = GeneralizedLinearModel(family="poisson",l2=.07,tolerance=1e-9).fit(X,y,sample_weight=weights)
    independent = PoissonRegressor(alpha=.07,tol=1e-10,max_iter=1000).fit(X,y,sample_weight=weights)
    np.testing.assert_allclose(native.coef_,independent.coef_,atol=1e-6)
    np.testing.assert_allclose(native.predict(X),independent.predict(X),atol=1e-6)
