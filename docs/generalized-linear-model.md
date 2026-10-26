# Generalized linear models (GLM)

LogitLab 0.4 adds a C++17 implementation of **generalized** linear models, with a thin NumPy/SciPy wrapper. It models a conditional mean through a distribution family and its canonical link. This is distinct from multivariate general linear regression.

| `family` | `link` (`"auto"` selects this) | Response | Mean μ = g⁻¹(η) | Variance function |
| --- | --- | --- | --- | --- |
| `gaussian` | `identity` | Finite real | η | 1 |
| `binomial` | `logit` | Finite values in [0, 1] | sigmoid(η) | μ(1−μ) |
| `poisson` | `log` | Finite nonnegative values | exp(η) | μ |

Only these pairs are implemented. Explicitly requesting another family or a noncanonical link raises an error. Nonnegative fractional Poisson responses can be used for quasi-likelihood mean estimation; they are not claimed to be integer count observations. Binomial proportions can be fit with trial counts supplied as sample weights. Response-domain checks apply to all rows, including zero-weight rows.

## Objective and row metadata

For nonnegative weights a with positive mass, qᵢ = aᵢ / Σa, and

$$\eta_i = x_i^\top w + b + o_i,\qquad
F(w,b)=\sum_i q_i\ell(y_i,\eta_i)+\frac{\lambda}{2}\|w\|_2^2.$$

The intercept is unpenalized; disabling it fixes b = 0. The per-observation loss omits constants depending only on the response:

- Gaussian: `(η−y)² / 2` (unit dispersion).
- Binomial: `softplus(η) − yη`, evaluated using stable signed softplus branches.
- Poisson: `exp(η) − yη` (the `log(y!)` term is omitted).

`l2` is λ in this **mean-loss** convention. Multiplying all weights by a positive constant leaves the objective and solution unchanged, including at positive λ. The core divides by the maximum weight before normalizing to avoid overflow. Weights are not used to estimate a sampling covariance, dispersion or degrees of freedom.

An `offset` is an additive, fixed contribution to η, not a fitted coefficient. Python additionally accepts strictly positive `exposure` **only for Poisson/log**, adding `log(exposure)` to the offset. Predictions are expected event **counts** at the supplied exposure, not exposure-normalized rates. Combine offset and exposure when needed. C++ callers supply the combined log exposure as an offset vector.

Training offsets/exposures are not retained. Supply them anew for prediction and deviance; omitted metadata means zero offset / unit exposure. This prevents accidentally reusing training-row metadata for unrelated prediction rows.

## C++ usage

```cpp
#include "logitlab/generalized_linear_model.hpp"

logitlab::Dataset data;
data.rows = 4;
data.cols = 1;
data.values = {0., 1., 2., 3.}; // owned, row-major; CSR is supported too
data.labels = {1., 2., 4., 7.};
logitlab::GLMOptions options;
options.family = "poisson";
options.link = "log";
options.l2 = .01;
auto result = logitlab::fit_glm(data, options);
auto expected_counts = logitlab::predict_glm(data, result);
double mean_deviance = logitlab::glm_deviance(data, result);
// Optional row vectors: fit_glm(data, options, sample_weight, offset).
```

Link `logitlab::core`. The public header exposes `GLMOptions`, `GLMResult`, `GLMIteration`, `fit_glm`, `predict_glm`, `glm_deviance` and `glm_objective`. The latter accepts `[w..., b]` and an output gradient, including a zero final component when the intercept is disabled. Dataset structure, dimensions, finite values, row metadata and response domains are validated by entry points.

Runnable examples:

```bash
cmake --preset release
cmake --build --preset release --parallel 2
./build/release/logitlab_glm_train
python examples/generalized_linear_model.py
```

## Python interface

```python
from logitlab import GeneralizedLinearModel

model = GeneralizedLinearModel(
    family="poisson", link="auto", l2=.01, fit_intercept=True,
    max_iter=100, max_cg_iter=200, tolerance=1e-7,
)
model.fit(X_train, y_train, sample_weight=weights, offset=offset_train,
          exposure=exposure_train)
expected_counts = model.predict(X_test, offset=offset_test, exposure=exposure_test)
eta = model.decision_function(X_test, offset=offset_test, exposure=exposure_test)
mean_deviance = model.deviance(X_test, y_test, offset=offset_test,
                              exposure=exposure_test)
```

`predict` always returns the conditional mean, including probabilities for Binomial; it does not threshold probabilities into class labels. Binomial additionally supports `predict_proba` returning columns `[P(0), P(1)]`. Other families reject that method. `decision_function` returns η. Sample weights are also accepted by `deviance`.

Fitted inspection attributes: `coef_`, `intercept_`, `family_`, `link_`, `n_features_in_`, `n_samples_`, `converged_`, `status_`, `history_` and `work_`. Histories include iteration, penalized loss, full-gradient norm, core seconds and all work counters. Inspection arrays and subsequently edited options do not alter the owned C++ fitted model. A failing refit preserves the previous model. Input conversion owns dense/CSR storage; training and prediction release the GIL.

## Solver and numerical behavior

Canonical links have gradient `Xᵀ(q*(μ−y)) + λw` and Hessian `Xᵀdiag(q*V(μ))X + diag(λ,...,λ,0)` with an intercept column when enabled. The implementation uses **Newton-CG with Jacobi preconditioning** and an Armijo backtracking line search. This is the canonical IRLS/Fisher-scoring Newton system, solved without forming its weighted design or Hessian. Inner CG runs at most `min(p+1, max_cg_iter)` steps and uses an adaptive residual target. A diagonal gradient direction is a fallback when curvature fails to produce a finite descent direction.

Each evaluation computes loss, gradient, row curvature and preconditioner in one data pass. Curvature is reused for the inner Hessian-vector products. Input storage is O(np) dense or O(nnz+n) CSR; additional working storage is O(n+p), plus O(T) history. Kernels are single-threaded and in-memory. The implementation has no new C++ library dependency and retains Abseil as its foundation. It does not silently scale input features; fit scaling on training data only.

Poisson initialization uses a weighted log-sum-exp of offsets so even a constant offset of 1000 can be compensated by an intercept without evaluating `exp(1000)`. Zero-weight rows are skipped before dot products, exponentials and curvature arithmetic. Non-finite trial steps are rejected and backtracked; η is not clipped to an arbitrary range. Non-finite initialization or prediction raises an error requesting input scaling rather than returning NaN/Inf. Very small Poisson means may underflow to zero on prediction; the loss/deviance still uses finite η directly where representable.

Stopping checks the full penalized gradient L2 norm against `tolerance`. Statuses distinguish `converged`, `max_iter`, `line_search_failed` and `numerical_breakdown`; the latter three return the last finite accepted iterate. Histories include the initial iterate (iteration 0). Work records full evaluations including rejected trials, Hessian-vector products, row visits and accepted parameter updates. A partial evaluation ending at a non-finite η counts only visited rows. Failed line-search work may exceed the final history entry's counters.

## Deviance and limitations

`glm_deviance` / `deviance` returns weighted **mean** unit deviance, excluding L2:

- Gaussian: `(y−μ)²`, so it equals weighted MSE.
- Binomial: `2[y log(y/μ) + (1−y) log((1−y)/(1−μ))]` with continuous boundary limits.
- Poisson: `2[y log(y/μ) − y + μ]`, with the `y=0` limit handled separately.

The implementation computes deviance from η, so Binomial probabilities rounding to exactly 0/1 do not turn finite extreme-margin losses into artificial infinities. Poisson deviance uses a local series near its minimum to reduce cancellation. Numerical negative roundoff is clamped to zero. Unlike a response-only metric, this method requires X and a fitted model.

This release does not estimate dispersion, standard errors, p-values, confidence intervals, overdispersion or zero-inflation. Gamma/Tweedie/negative-binomial families, noncanonical links, L1 and coefficient constraints are not part of this GLM solver. Existing constrained least squares remains available through `LinearRegression`.

An intercept-only all-zero Poisson response or all-one/all-zero Binomial response has no finite unpenalized intercept MLE; complete separation has similar issues. A small gradient can be attained along such a diverging coefficient path. `converged` certifies the numerical gradient criterion, not existence/uniqueness of a finite MLE or validity of inference. Rank-deficient designs may also have nonunique coefficients. Compare fitted means/objectives and inspect coefficients/status in these cases.

## TDD and validation

The native test first failed because `generalized_linear_model.hpp` did not exist; the Python test then failed because the installed baseline did not export `GeneralizedLinearModel`. Implementation followed these RED checks.

- Native finite-difference gradients cover every family with offsets, nonuniform/zero weights and L2.
- Python tests compare weighted solutions to independently formulated SciPy BFGS problems for every family, dense/CSR and with/without intercept. Poisson is additionally checked against scikit-learn `PoissonRegressor`; Gaussian/Binomial agree with the existing logits estimators.
- Further contracts cover deviance formulas, proportions/trial weights, exposure equivalence, scale-invariant weights, explicit response/link validation, failed refits, immutable prediction state, budget status, zero-weight overflow, saturated Binomial margins and Poisson overflow backtracking.

```bash
python -m pytest -q tests/test_glm.py
ctest --test-dir build/release --output-on-failure
```

These are correctness checks, not a claim that Newton-CG is universally fastest. The existing benchmark reports predate this GLM solver and do not measure Poisson throughput.

## 中文速读

这里实现的是**广义线性模型**：Gaussian/identity、Binomial/logit、Poisson/log。C++ 统一求解带权重和 L2 的均值负对数似然，Python 只负责接口与 exposure → log offset 转换。采用矩阵无关 Newton-CG，支持 dense/CSR，不构造稠密 Hessian。

`predict` 返回条件均值：高斯为实数、二项为概率、泊松为指定 exposure 下的期望计数。offset/exposure 需要在预测时重新提供；`deviance` 是带权平均偏差，不含正则项。样本权重整体缩放不影响解。检查 `status_` 和完整梯度范数，不把预算耗尽等同于收敛。

首版仅支持上述规范 link，尚未加入其他分布族、非规范 link、GLM 参数约束或统计显著性推断。原有线性回归的约束与分析能力继续通过 `LinearRegression` 提供。上面的 C++/Python 示例、公式和验证命令可直接用于复现。
