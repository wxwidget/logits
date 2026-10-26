"""Poisson events with varying exposure, held-out mean deviance and CSR input."""
import numpy as np
from scipy import sparse
from logitlab import GeneralizedLinearModel

rng = np.random.default_rng(42)
X = rng.normal(size=(1000, 4))
exposure = rng.uniform(.5, 3., len(X))
rate = np.exp(X @ [.3, -.2, .1, .4] + .2)
y = rng.poisson(exposure * rate).astype(float)
train, test = slice(0, 800), slice(800, None)
model = GeneralizedLinearModel(family="poisson", l2=.01, tolerance=1e-8)
model.fit(sparse.csr_matrix(X[train]), y[train], exposure=exposure[train])
prediction = model.predict(sparse.csr_matrix(X[test]), exposure=exposure[test])
print("Poisson:", model.status_, "coefficients:", model.coef_)
print("Held-out mean deviance:", model.deviance(X[test], y[test], exposure=exposure[test]))
print("First expected counts:", prediction[:5])
assert model.converged_ and np.isfinite(prediction).all() and np.all(prediction > 0)

probability = 1 / (1 + np.exp(-(X[:, 0] - .2)))
binary = rng.binomial(1, probability).astype(float)
logistic = GeneralizedLinearModel(family="binomial", l2=.01).fit(X[train], binary[train])
print("Binomial:", logistic.status_, "probabilities:", logistic.predict_proba(X[test])[:3])
assert logistic.converged_

continuous = X @ [.4, -.2, 0., .1] + .3 + rng.normal(scale=.2, size=len(X))
gaussian = GeneralizedLinearModel(family="gaussian").fit(X[train], continuous[train])
print("Gaussian:", gaussian.status_, "test MSE:", gaussian.deviance(X[test], continuous[test]))
assert gaussian.converged_
