# Python API and numerical contract

```python
from logitlab import LogisticRegression
model = LogisticRegression(solver="lbfgs", l2=0.1).fit(X, y)
probabilities = model.predict_proba(X_test)
labels = model.predict(X_test)
```

## Input and output

`fit(X, y)` returns `self`. Dense input is converted to row-major float64. `X` must have shape `(n_samples, n_features)` with both dimensions positive; `y` must be one-dimensional with one exact 0/1 label per row. NaN/Inf features and labels are rejected. SciPy sparse matrices/arrays are copied, checked and converted to canonical CSR: duplicates summed and indices sorted. Inputs are not modified and sparse input is not densified.

`predict_proba(X)` returns shape `(n_samples, 2)`, columns `[P(y=0), P(y=1)]`. `predict(X)` returns int64 labels using `P(y=1) >= 0.5`. Prediction requires a fitted model and the training feature count. Empty prediction input is rejected. The wrapper is a small research API, not a complete scikit-learn estimator; arbitrary label encoding, sample weights, multiclass training, model serialization and sklearn parameter-search methods are not implemented.

The wrapper owns a copy of float64 feature storage. Training and prediction release the GIL; training snapshots options before release. Exposed `coef_`, `intercept_` and history are inspection copies: assigning them does not change the C++ prediction model. A failed refit leaves the previous fitted model available.

## Constructor

| Parameter | Default | Meaning |
| --- | --- | --- |
| `solver` | `"lbfgs"` | `gd`, `sgd`, `adam`, `lbfgs`, `newton-cg` |
| `max_epochs` | `100` | Positive epoch/outer-iteration budget |
| `learning_rate` | `0.05` | Positive finite step for GD/SGD/Adam; GD backtracks |
| `batch_size` | `256` | Positive maximum stochastic mini-batch size |
| `l2` | `1e-4` | Finite nonnegative coefficient in `l2/2 * ||w||²` |
| `tolerance` | `1e-6` | Positive finite full-gradient L2 norm threshold |
| `history_size` | `10` | Positive maximum L-BFGS curvature-pair count |
| `seed` | `42` | Nonnegative 64-bit shuffle seed |
| `fit_intercept` | `True` | Train an unpenalized intercept; otherwise it is zero |

Batch size and history size must be positive even for solvers that do not use them. A batch size larger than the data uses a full batch. Adam uses beta1=0.9, beta2=0.999, epsilon=1e-8; stochastic methods use a constant step size. Newton-CG caps each inner solve at `min(p+1, 50)` iterations and damps its direction system by `1e-10`; the reported objective is unchanged.

## Fitted attributes

- `coef_`: one float64 weight per feature.
- `intercept_`: scalar intercept.
- `converged_`: true only when the recorded full gradient meets tolerance.
- `status_`: `converged`, `max_epochs` or `line_search_failed`.
- `history_`: dictionaries containing `epoch`, penalized mean `loss`, `gradient_norm`, and C++ `seconds` since fit started. Entry zero evaluates the initial all-zero model. Subsequent entries record accepted outer steps or completed stochastic epochs.
- `work_`: cumulative `full_evaluations` (joint objective/full-gradient evaluations), `batch_gradients`, `hessian_vector_products`, `row_visits`, and `parameter_updates`. Every history entry also snapshots these counters. Totals include rejected line-search evaluations after the last recorded step.

History adds a full data pass after every stochastic epoch. Timing starts after C++ input validation and optimizer workspace allocation; it excludes Python input conversion. Seed reproducibility is tested on the same toolchain; shuffle order can differ across standard-library implementations.

A row visit is one objective/gradient, mini-batch, curvature or Hessian-vector traversal of a sample. Dividing by the training row count gives equivalent data passes. This is a work proxy, not FLOPs: Newton curvature rows only evaluate scores, other traversals also accumulate vectors, and dense parameter-vector operations are not counted in row visits. Parameter updates count accepted batch steps or stochastic mini-batch steps; a failed line search adds evaluations but no update.

Invalid numeric values, shapes, labels, solver names or zero budgets raise `ValueError`. Negative integer arguments can also be rejected by pybind11 during conversion. Prediction before fitting, non-finite optimization progress and undefined prediction scores raise `RuntimeError`. Infinite prediction scores saturate to 0/1; undefined scores from floating-point cancellation are rejected. A finite dataset with extremely large values can still overflow arithmetic; scale features rather than accepting non-finite progress.

## C++ API

Include `logitlab/logistic_regression.hpp` and link `logitlab::core`. `Dataset` owns its vectors; dense values are row-major, CSR uses `offsets` and sorted unique `indices`. All public fit/predict/objective entry points validate input. Low-level `Dataset::dot/add` assume validated storage, an in-range row and correctly sized spans.

`objective(data, parameters, options, gradient)` takes `p+1` parameters `[weights..., intercept]` even when intercept fitting is disabled, and a separate output span of the same size. Its gradient output must not alias the parameters. It returns penalized mean log loss, and validates finite parameters/nonnegative finite L2. Fit validates once and uses the internal evaluator inside iterations, avoiding repeated dataset scans. C++ invalid inputs throw `std::invalid_argument`; invalid numerical progress throws `std::runtime_error`.
