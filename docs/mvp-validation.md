# MVP acceptance — 2026-10-14

This record describes local validation of LogitLab 0.2.0 for the `wxwidget/logits` master update. GitHub Actions is configured for additional Python versions/compiler jobs; a configured job is not evidence of a remotely completed run.

## Environment

Linux x86_64; GCC 14.2.0; CMake 4.4.4; Python 3.12.14; NumPy 2.5.4; SciPy 1.18.1. Portable Release uses normal optimization without fast-math or CPU-specific instructions. C++ kernels are single-threaded.

The restored cloud machine preserved the checkout at `/workspace/regressionlib` and its dependency caches. The reusable helper resolves its own repository directory; it also works when the checkout is named `logits`. Installed dependencies and fixed FetchContent sources were reused to build/install/test without network inside the execution sandbox. Git access to the canonical `https://github.com/wxwidget/logits.git` worked through the existing platform proxy outside the network-restricted sandbox. A GitHub API token was not needed.

## Acceptance checks

- 76 Python tests: independent SciPy solution checks and finite-difference gradients; all five solvers; dense/CSR agreement; budgets, convergence and line-search outcomes; numerical divergence; invalid shape/data/options; malformed CSR/CSC indices; duplicates and input preservation; dtype/layout conversions; reproducibility; intercept/L2 conventions; failed refit behavior.
- Python wrapper coverage: 100% of 37 statements. This is not a C++ coverage percentage.
- Release CTest: core numerical/storage contract checks and the runnable C++ example (two registered tests).
- AddressSanitizer, UndefinedBehaviorSanitizer and LeakSanitizer: the same two C++ targets, with leak checking enabled and UBSan configured to halt. LeakSanitizer initially failed under the sandbox's ptrace restriction; the checks were rerun outside that restriction rather than disabling leak detection.
- Runnable dense/CSR Python example: both converged and predicted `[0 0 1 1]`; C++ example converged on the same data.
- Environment helper: repeated setup reuses dependencies, rebuilds/installs the package, checks dependency consistency and executes its readiness checks.

- Source packaging: the generated `logitlab-0.2.0.tar.gz` was used to build a wheel, which was installed in a separate venv. From `/tmp`, both the Python package and compiled extension were confirmed to load from that venv, and dense/CSR training/prediction were exercised. The isolated venv reused system runtime dependencies; it did not import LogitLab from the development venv or checkout.
- Historical code: all seven opt-in executables built; LR and L-BFGS completed on the bundled 1000-row CSV data, and FTRL produced 270 predictions. These smoke checks do not establish modern-objective equivalence.
- Relative links in README and the Markdown documentation resolved to existing files. Historical benchmark files were retained unchanged.

## Reproduce

```bash
bash tools/setup_environment.sh
python -m pytest -q --cov=logitlab --cov-report=term-missing --cov-fail-under=95
```

See [development](development.md) for sanitizer/source-package commands and [initial benchmark evidence](validation.md) for historical synthetic runs. Those initial timings precede the latest input validation/allocation changes and should not be presented as measurements of every subsequent revision.
