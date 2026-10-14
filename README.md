# LogitLab

**Logistic regression optimization research: a C++17 core with a NumPy/SciPy interface.**

[Quick start](#quick-start) · [Python API](docs/api.md) · [C++ architecture](docs/architecture.md) · [Benchmarks](docs/benchmarking.md) · [Development](docs/development.md)

LogitLab compares optimizers using the same binary logistic loss, regularization and dense/CSR kernels. It reports loss, full-gradient norm, elapsed time and termination status so you can study how an algorithm reaches a solution and how that solution performs on held-out data.

The repository is named **logits**; the Python package and C++ namespace are **logitlab**. Version 0.2.0 is the first modern MVP. Kernels currently run on one CPU thread and training holds the dataset in memory.

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

## One objective, five optimizers

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

```bash
python benchmarks/compare.py --rows 10000 --cols 100 --epochs 30
python benchmarks/compare.py --rows 100000 --cols 10000 --density 0.0005 \
  --epochs 10 --solvers sgd adam lbfgs newton-cg --output build/sparse-benchmark
```

The runner uses seeded synthetic data, an 80/20 train/validation split and training-only RMS scaling. JSON records size/nonzeros, runtime metadata, fit time, objective, gradient norm and held-out log loss/accuracy; CSV records convergence traces. Step size, batch size, L2 and tolerance are configurable.

Compare time to the same target objective or gradient tolerance across several seeds. Equal iteration budgets involve different amounts of work. Defaults are research baselines, not equally tuned configurations. Fit time includes Python conversion; trace time measures C++ optimization. See [the protocol](docs/benchmarking.md) and [retained initial results](docs/validation.md), including poor Adam results and unconverged runs. These are synthetic smoke benchmarks, not a claim of peak throughput.

## Validation and contribution

The suite checks independently computed gradients and SciPy optima, all five solvers, dense/CSR agreement, invalid storage/parameters, extreme margins, convergence status, intercept semantics and seeded training. C++ checks remain active in Release builds; sanitizer and source-package/wheel jobs are configured in CI. [MVP acceptance record →](docs/mvp-validation.md)

Use [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow. Performance changes should include reproducible convergence traces and numerical checks. [CHANGELOG.md](CHANGELOG.md) describes the MVP; streaming, multicore kernels, lazy sparse updates and further variance-reduction solvers remain future work.

## Historical code and licensing

Opt-in historical executables retain their original objectives and conventions. `python/` standalone scripts and `pytron/` are archived experiments; the supported package is `python/logitlab/`. See [legacy development](docs/development.md#historical-experiments).

This repository has no declared project-wide license. No new license grant is inferred. Third-party TRON notices are in [pytron/COPYING](pytron/COPYING); Abseil and pybind11 retain their upstream licenses.
