# LogitLab

**Linear and logistic regression research: a C++17 core with a NumPy/SciPy interface.**

[Quick start](#quick-start) · [Linear regression](docs/linear-regression.md) · [Python API](docs/api.md) · [Benchmarks](docs/benchmarking.md) · [Development](docs/development.md)

LogitLab studies model regularization and optimization on shared dense/CSR kernels. Binary logistic regression has five optimizers; linear regression adds OLS/WLS, Ridge, Lasso, Elastic Net, explicit coefficient constraints and OLS analysis. Histories, numerical references and held-out benchmarks connect optimization progress with predictive quality.

The repository is named **logits**; the Python package and C++ namespace are **logitlab**. Version **0.3.0** adds linear regression and analysis. Kernels currently run on one CPU thread and training holds the dataset in memory.

## Quick start

Requires Python 3.10+, Git and a C++17 compiler. The Python build installs CMake/Ninja and fetches fixed Abseil and pybind11 commits. First-time installation needs access to PyPI and GitHub. Linux is the currently verified platform.

```bash
git clone https://github.com/wxwidget/logits.git
cd logits
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[test]'
python -m pytest -q
```

```python
import numpy as np
from logitlab import LogisticRegression

X = np.array([[-2., 0.], [-1., 1.], [1., 0.], [2., 1.]])
y = np.array([0., 0., 1., 1.])
model = LogisticRegression(solver="lbfgs", l2=0.1).fit(X, y)
print(model.predict(X))           # [0 0 1 1]
print(model.predict_proba(X))     # columns: P(y=0), P(y=1)
print(model.status_)             # converged
print(model.history_[-1])        # epoch, loss, gradient_norm, seconds
```

Use a SciPy sparse matrix through the same API:

```python
from scipy.sparse import csr_matrix
sparse_model = LogisticRegression(l2=0.1).fit(csr_matrix(X), y)
```

Inputs must be finite, two-dimensional and nonempty, with binary labels exactly 0/1. Sparse inputs are checked and canonicalized to CSR without densification. Features should be scaled using training data only; avoid centering sparse features. See [the walkthrough](docs/getting-started.md) for complete dense, sparse and C++ examples.

## Linear models, constraints and analysis

```python
import numpy as np
from logitlab import LinearRegression, Ridge, Lasso, ElasticNet

rng = np.random.default_rng(42)
X = rng.normal(size=(400, 4))
y = X @ [2., -1., 0., .5] + 3 + rng.normal(scale=.5, size=400)
X_train, X_test = X[:320], X[320:]
y_train, y_test = y[:320], y[320:]
linear = LinearRegression().fit(X_train, y_train)
print(linear.evaluate(X_test, y_test))  # MSE, RMSE, MAE, R², explained variance
ridge = Ridge(alpha=0.01).fit(X_train, y_train)
sparse_model = Lasso(alpha=0.01).fit(X_train, y_train)
elastic = ElasticNet(alpha=0.01, l1_ratio=0.5).fit(X_train, y_train)
bounded = LinearRegression(
    positive=True, bounds=(0., 2.), fixed_coefficients={2: 0.}
).fit(X_train, y_train)
```

Linear models minimize weighted mean squared loss/2 plus L1 and L2 penalties; the intercept is unpenalized. Matrix-free preconditioned CG handles OLS/Ridge. Proximal gradient and monotone restarted FISTA handle L1, nonnegative, box and fixed coefficients. All support dense/CSR inputs; weights are passed through fit(sample_weight=...).

For a converged unregularized/unconstrained dense OLS fit, linear.summary(X_train, y_train, covariance="HC3") provides coefficient uncertainty, confidence intervals, adjusted R², AIC/BIC and influence diagnostics. Inference checks rank, degrees of freedom and fit stationarity; it does not apply ordinary OLS significance tests to penalized/constrained models. [Objective conventions, constraints and examples →](docs/linear-regression.md)

The work followed **TDD**, including observed failing tests for the new API, solver precision and numerical edge cases. [RED/GREEN record →](docs/tdd-linear-regression.md)

## Logistic regression: one objective, five optimizers

Every modern solver minimizes

$$F(w,b)=\frac{1}{n}\sum_{i=1}^{n}\left[\log(1+e^{x_i^\top w+b})-y_i(x_i^\top w+b)\right]+\frac{\lambda}{2}\lVert w\rVert_2^2.$$

`l2` is $\lambda$. Loss is averaged over samples and the intercept is unpenalized. Stable softplus and sigmoid branches handle extreme margins. The same objective and gradient are used by every solver.

| Solver | Update | Useful research question | Additional working memory |
| --- | --- | --- | --- |
| `gd` | Full gradient with Armijo backtracking | How does a simple batch baseline converge? | $O(p)$ |
| `sgd` | Shuffled mini-batch gradient | How do batch size, step size and seed affect training? | $O(n+p)$ |
| `adam` | Mini-batch adaptive moments | When do adaptive steps help or hurt? | $O(n+p)$ |
| `lbfgs` | Limited-memory two-loop direction and backtracking | How much does curvature history help? | $O(mp)$ |
| `newton-cg` | Truncated CG, Hessian-vector products and backtracking | What accuracy is gained per extra data pass? | $O(n+p)$ |

Here $n$ is the row count, $p$ the feature count and $m$ the L-BFGS history size. Dataset storage is additional: $O(np)$ dense or $O(\mathrm{nnz}+n)$ CSR. Newton-CG does not form a dense Hessian. CSR mini-batches still perform $O(p)$ regularization/adaptive updates. Returned history adds $O(T)$ storage for $T$ recorded steps.

`max_epochs` counts epochs for stochastic solvers and outer iterations for batch solvers. `converged` means full-gradient L2 norm is at most `tolerance`; `max_epochs` and `line_search_failed` are separate outcomes. Constant-step SGD and Adam can exhaust their budget without convergence. Check `status_` and `history_`, not just a successful process exit. [Defaults and contracts →](docs/api.md)

## C++ development

```bash
python -m pip install cmake ninja
cmake --preset release
cmake --build --preset release --parallel 2
ctest --preset release
./build/release/logitlab_train
```

```cpp
#include "logitlab/logistic_regression.hpp"
// Fill a logitlab::Dataset with owned dense or CSR storage and 0/1 labels.
logitlab::Options options;
options.solver = "lbfgs";
auto model = logitlab::fit(data, options);
auto probabilities = logitlab::predict(data, model);
```

Link the `logitlab::core` CMake target. The [complete C++ example](examples/train.cpp) is built and checked by CTest. Abseil provides span views and historical parsing/container utilities; Boost is no longer required. [Build flags, offline setup and cloud recovery →](docs/development.md)

## Reproducible experiments

**Kaggle real-data workflow:** [中文指南](docs/kaggle-benchmark.zh-CN.md) · [English guide](docs/kaggle-benchmark.md). Breast Cancer examines numerical scaling/correlation, Adult examines imbalanced one-hot CSR classification, and Insurance compares linear regularization. The downloader records observed versions/licenses and hashes; the runner produces bilingual quality/performance reports. **Actual Kaggle measurements are pending cloud network allowlist activation; the offline protocol has been tested.**

```bash
python -m pip install '.[test,benchmark]'
python benchmarks/kaggle_data.py
python benchmarks/kaggle_suite.py --output build/kaggle-benchmark
python benchmarks/kaggle_report.py --input build/kaggle-benchmark --output build/kaggle-report
```

The **linear regression study** selects regularization on validation data, compares CG/proximal GD/FISTA against independent optimal references, and measures prediction error, coefficient recovery, constraint feasibility, training cost and memory:

```bash
python -m pip install '.[test,benchmark]'
python benchmarks/linear_suite.py --output build/linear-benchmark
python benchmarks/linear_report.py --input build/linear-benchmark --output build/linear-report
```

[线性回归中文报告](benchmarks/reports/2026-10-16-linear/linear.zh-CN.md) · [English regression report](benchmarks/reports/2026-10-16-linear/linear.en.md)

The separate **logistic optimizer study** remains reproducible:

`benchmarks/suite.py` evaluates all five optimizers on comparable-scale, multiscale, correlated, high-dimensional CSR and bundled real-data scenarios. It compares raw features with training-only RMS scaling in separate unregularized and L2 tracks, calibrates step sizes using training objective, and records repeated fit/prediction measurements, independent SciPy objective gaps, actual data passes and isolated-worker peak RSS.

```bash
python -m pip install '.[test,benchmark]'
python benchmarks/suite.py --output build/benchmark-suite
python benchmarks/report.py --input build/benchmark-suite --output build/benchmark-report
```

Read the measured **[中文评测报告](benchmarks/reports/2026-10-15/benchmark.zh-CN.md)** or **[English report](benchmarks/reports/2026-10-15/benchmark.en.md)**, with scaling/convergence charts, predictive quality, performance tables and batch/history ablations. Raw results and traces are retained alongside the reports. These compare optimization of the same model family; scaling with fixed L2 also changes the effective regularization.

For a quick protocol check, use `--quick --cases dense_control --seeds 42 --epochs 2 --repeats 1`. Quick runs verify execution and do not support throughput claims.

The original configurable synthetic smoke runner remains available:

```bash
python benchmarks/compare.py --rows 10000 --cols 100 --epochs 30
python benchmarks/compare.py --rows 100000 --cols 10000 --density 0.0005 \
  --epochs 10 --solvers sgd adam lbfgs newton-cg --output build/sparse-benchmark
```

The runner uses seeded synthetic data, an 80/20 train/validation split and training-only RMS scaling. JSON records size/nonzeros, runtime metadata, fit time, objective, gradient norm and held-out log loss/accuracy; CSV records convergence traces. Step size, batch size, L2 and tolerance are configurable.

Compare time to the same target objective or gradient tolerance across several seeds. Equal iteration budgets involve different amounts of work. Defaults are research baselines, not equally tuned configurations. Fit time includes Python conversion; trace time measures C++ optimization. See [the protocol](docs/benchmarking.md) and [retained initial results](docs/validation.md), including poor Adam results and unconverged runs. These are synthetic smoke benchmarks, not a claim of peak throughput.

## Validation and contribution

The suite checks independent logistic/linear gradients and optima, dense/CSR agreement, weighted fits, L1 closed forms, parameter bounds, classical/HC3 inference, numerical edge cases and benchmark split/selection rules. C++ checks remain active in Release builds; sanitizer and source-package/wheel jobs run in CI. [Linear TDD record →](docs/tdd-linear-regression.md) · [Original MVP acceptance →](docs/mvp-validation.md)

Use [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow. Performance changes should include reproducible convergence traces and numerical checks. [CHANGELOG.md](CHANGELOG.md) describes the MVP; streaming, multicore kernels, lazy sparse updates and further variance-reduction solvers remain future work.

## Historical code and licensing

Opt-in historical executables retain their original objectives and conventions. `python/` standalone scripts and `pytron/` are archived experiments; the supported package is `python/logitlab/`. See [legacy development](docs/development.md#historical-experiments).

This repository has no declared project-wide license. No new license grant is inferred. Third-party TRON notices are in [pytron/COPYING](pytron/COPYING); Abseil and pybind11 retain their upstream licenses.
