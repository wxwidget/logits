# Linear regression and parameter constraints

LogitLab 0.3 adds OLS, weighted least squares, Ridge, Lasso and Elastic Net. Training and prediction run in C++17 on owned float64 dense/CSR data. The Python layer converts input, exposes models and computes optional regression statistics with NumPy/SciPy.

## Objective and model choices

For nonnegative sample weights with positive total mass:

$$F(w,b)=\frac{\sum_i a_i(y_i-x_i^\top w-b)^2}{2\sum_i a_i}
+\lambda_1\|w\|_1+\frac{\lambda_2}{2}\|w\|_2^2.$$

The intercept is unpenalized. Missing weights mean equal weights; zero-weight rows are excluded from numerical kernels. Weight normalization avoids overflow and preserves the answer when all weights are multiplied by a common positive factor.

| Python model | Parameters in this objective | Purpose |
| --- | --- | --- |
| LinearRegression() | l1=0, l2=0 | Ordinary least squares |
| LinearRegression().fit(..., sample_weight=a) | Weighted squared loss | WLS; weights represent relative observation precision for inference |
| Ridge(alpha=α) | l1=0, l2=α | L2 shrinkage |
| Lasso(alpha=α) | l1=α, l2=0 | Sparse coefficients |
| ElasticNet(alpha=α, l1_ratio=ρ) | l1=αρ, l2=α(1−ρ) | Combined sparsity and shrinkage |

Ridge uses **mean squared loss**, so its α differs from scikit-learn Ridge's sum-loss α. With equal weights and n rows, sklearn Ridge needs alpha=n*α to match this objective. Lasso/Elastic Net follow the mean-loss convention directly. Named models reject conflicting regularizers; use LinearRegression(l1=..., l2=...) for an explicit combination.

~~~python
import numpy as np
from logitlab import LinearRegression, Ridge, Lasso, ElasticNet

rng = np.random.default_rng(42)
X = rng.normal(size=(400, 4))
y = X @ [2., -1., 0., .5] + 3 + rng.normal(scale=.5, size=400)
model = LinearRegression().fit(X, y)
print(model.predict(X[:3]))
print(model.evaluate(X, y))  # mse, rmse, mae, r2, explained_variance
print(model.status_, model.solver_, model.work_)
~~~

SciPy sparse matrices/arrays use the same API and are canonicalized to CSR without densification. The estimator never scales features automatically. Fit a scaler on training rows only; avoid centering sparse inputs.

## Solvers and convergence

| Solver | Supported objectives | Behavior |
| --- | --- | --- |
| auto (default) | All | CG for smooth unconstrained fits; FISTA otherwise |
| cg | l1=0, no active coefficient/intercept bounds | Matrix-free conjugate gradient with Jacobi preconditioning; no XᵀX or dense Hessian |
| proximal-gd | All | Soft thresholding plus box projection, with backtracking |
| fista | All | Accelerated proximal steps with monotone objective restart and backtracking |

Defaults: max_iter=1000, tolerance=1e-7, learning_rate=1, fit_intercept=True. Learning rate is a positive **initial/maximum** proximal step; backtracking reduces it and successful updates may grow it back. CG accepts the common parameter but does not use it.

All solvers start with zero coefficients projected onto the feasible set and a feasible weighted-mean intercept. For L1/box constraints, a raw gradient need not vanish at an optimum. The stationarity criterion is:

$$\left\|\theta-\operatorname{prox}_{\lambda_1\|w\|_1+I_C}(\theta-\nabla f(\theta))\right\|_2
\le\text{tolerance}.$$

The proximal mapping uses unit step and does not penalize the intercept. In the smooth unconstrained case it equals the full-gradient norm. It is coordinate-dependent; compare optimization time to a matched objective gap when studying scaling.

FISTA backtracking checks the quadratic curvature inequality using gradient differences, avoiding loss-subtraction cancellation near a high-precision solution. Acceleration points may lie outside the box; every accepted iterate and returned coefficient remains feasible.

Status is converged, max_iter, line_search_failed or numerical_breakdown. Budget exhaustion returns the current model with converged_=False. Non-finite progress raises RuntimeError. CG can return different coefficients in a rank-deficient problem than an SVD minimum-norm solution; compare predictions/objective since coefficients are not unique.

history_ records iteration, loss, stationarity_norm, seconds and the same work-counter names as logistic regression. work_ totals include rejected trials after the last recorded iterate. Row visits include CG diagonal construction/HVPs and every objective/full-gradient evaluation, including proximal anchor evaluations. They are a data-pass proxy, not FLOPs. CG uses O(p) optimizer vectors; weighted fits add O(n) normalized weights; all solvers retain O(T) history. Input storage is additional.

## Coefficient and intercept constraints

~~~python
model = LinearRegression(
    bounds=(-2., 2.),                  # scalar or per-feature lower/upper arrays
    fixed_coefficients={2: 0.},        # index -> exact coefficient
    intercept_bounds=(1., 4.),
    max_iter=3000,
).fit(X, y)

nonnegative = LinearRegression(positive=True).fit(X, y)
~~~

- bounds=(lower, upper): either endpoint can be None, scalar or a vector with one value per feature. None means unbounded; −∞/+∞ represent open endpoints.
- Equal finite endpoints fix a coefficient. fixed_coefficients is a convenience for the same condition; indices are zero-based integers.
- positive=True imposes w≥0 and intersects existing lower bounds. Fixed negative coefficients are rejected.
- intercept_bounds is a separate (lower, upper) pair, with optional equal finite endpoints to fix the intercept. It does not regularize the intercept.
- With fit_intercept=False, b=0 and any supplied intercept interval must contain zero.
- Contradictory bounds, NaN, lower=+∞, upper=−∞ or fixed values outside the interval raise ValueError.

General coupled constraints such as Aw≤c or Aw=c are not implemented. These methods enforce separable bounds; they do not solve a general quadratic program.

For the Python estimator, configure feature constraints through bounds/fixed_coefficients and intercept constraints through intercept_bounds. Fit constructs the low-level bound vectors from these settings. The exposed options object controls numeric solver options; direct C++ callers can set LinearOptions bound vectors.

Constraints apply in the **input feature coordinates**. If X_scaled=X/s with positive s, then w_scaled=s*w_original: transform bounds and fixed values by s too. Nonnegativity is invariant to positive RMS scaling. Scalar L1/L2 penalties after scaling represent different original-coordinate penalties.

## Metrics and residuals

~~~python
from logitlab import regression_metrics
prediction = model.predict(X)
metrics = regression_metrics(y, prediction)
residual = model.residuals(X, y)  # y - prediction
weighted_metrics = model.evaluate(X, y, sample_weight=np.linspace(.5, 2, len(y)))
~~~

Metrics use normalized weights. R² is centered on the weighted target mean, even for a no-intercept estimator; constant targets use 1 for exact predictions and 0 otherwise. score returns R². Explained variance removes the mean residual. Finite inputs are required, and overflowed metrics/predictions are rejected.

Exposed coef_/intercept_ are inspection copies; mutating them does not change prediction. Failed refits preserve the previous model. Options are copied before C++ releases the GIL. The model retains coefficients and fit metadata, not the training matrix.

## OLS/WLS analysis

~~~python
summary = LinearRegression().fit(X, y).summary(
    X, y, confidence=.95, covariance="HC3"
)
print(summary["coefficients"])  # estimate, SE, statistic, p-value, confidence interval
print(summary["adjusted_r2"], summary["aic"], summary["bic"])
~~~

The summary requires a converged, unpenalized, unconstrained least-squares fit. Supply the fitting data and, for WLS, the same sample weights; normal-equation stationarity is checked. Ridge/Lasso/Elastic Net and constrained fits can use evaluate/residuals, but do not receive ordinary OLS significance estimates.

Analysis uses SVD of the weighted design and never forms an n-by-n hat matrix. It provides coefficient covariance, standard errors, tests/intervals, rank, condition number, residual degrees of freedom, adjusted R², fitted values, residuals, leverage and Cook's distance.

- classical assumes independent errors with variances proportional to inverse weights; intervals/tests use Student-t. The classical overall F test is also returned.
- HC3 provides a heteroscedasticity-robust sandwich covariance and asymptotic normal tests/intervals. It does not correct serial dependence or clustering. Its classical F statistic is omitted.
- AIC/BIC use the Gaussian OLS/WLS likelihood with relative inverse-variance weights and count regression parameters (including intercept), matching the common regression-report convention. They are not penalized-model effective-degrees-of-freedom criteria.
- No-intercept summary R²/adjusted R² use an **uncentered** total sum of squares. This differs intentionally from the always-centered predictive score.
- Zero-weight observations are excluded from nobs, residual arrays and degrees of freedom. These are precision weights, not frequency weights.
- Sparse inference requires explicit dense input. The default feature cap is 512 (configurable with max_features); the design cap is five million elements. Prediction/metrics remain sparse.
- Rank deficiency, nonpositive residual degrees of freedom/variance, or unit leverage for HC3 reject inference. Cook's distance is NaN at unit leverage. A small p-value is conditional on model/error assumptions; it is not evidence of causation.

## C++ API

~~~cpp
#include "logitlab/linear_regression.hpp"
logitlab::LinearOptions options;
options.l1 = 0.01;
options.positive = true;
auto result = logitlab::fit_linear(data, options);
auto prediction = logitlab::predict_linear(data, result);
~~~

Link logitlab::core. Dataset labels are finite real values at linear entry points; logistic entry points still require 0/1. linear_objective takes p+1 feasible parameters and returns the penalized objective plus the **smooth-part** gradient. Optional sample weights must be empty or match row count. It validates shapes, options and feasibility before kernels. The output gradient must not alias parameters.

The [Python example](../examples/linear_regression.py) and [C++ example](../examples/linear.cpp) are exercised by tests/CI. See the [TDD record](tdd-linear-regression.md) and [benchmark protocol](benchmarking.md).
