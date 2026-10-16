"""OLS inference, regularization, sample weights and explicit parameter bounds."""
import numpy as np
from scipy import sparse
from logitlab import ElasticNet, LinearRegression, Ridge

rng = np.random.default_rng(42)
X = rng.normal(size=(400, 4))
y = X @ [2., -1., 0., .5] + 3 + rng.normal(scale=.5, size=len(X))
model = LinearRegression().fit(X, y)
print("OLS:", model.evaluate(X, y), model.status_)
summary = model.summary(X, y, covariance="HC3")
print("Coefficient analysis:", summary["coefficients"])
print("Ridge:", Ridge(alpha=.01).fit(sparse.csr_matrix(X), y).evaluate(X, y))
print("Elastic Net:", ElasticNet(alpha=.03, l1_ratio=.5).fit(X, y).coef_)
constrained = LinearRegression(bounds=(-2., 2.), fixed_coefficients={2: 0.}).fit(X, y)
assert constrained.coef_[2] == 0
assert np.all(abs(constrained.coef_) <= 2)
weights = np.linspace(.5, 2, len(y))
weighted = LinearRegression().fit(X, y, sample_weight=weights)
print("WLS:", weighted.evaluate(X, y, sample_weight=weights))
