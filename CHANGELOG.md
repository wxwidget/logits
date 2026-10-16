# Changelog

## 0.3.0 — Linear regression and analysis (2026-10-16)

- C++ OLS/WLS, Ridge, Lasso and Elastic Net with dense/CSR input, normalized sample weights and unpenalized intercept.
- Matrix-free Jacobi-preconditioned CG, proximal GD and monotone restarted FISTA with explicit stationarity/work histories.
- Nonnegative, per-feature box, fixed-coefficient and intercept bounds; invalid combinations rejected.
- Python regression metrics/residuals and restricted SVD-based OLS/WLS inference with classical/HC3 covariance, intervals and influence diagnostics.
- TDD solution/statistics/constraint/edge tests, C++/Python examples and regression model/optimizer benchmarks with validation-only penalty selection and bilingual reports.

## Unreleased — Optimizer research benchmark (2026-10-15)

- Reproducible five-scenario optimizer suite with raw/RMS and unregularized/L2 tracks, stratified splits, independent SciPy references and training-only step calibration.
- Repeated training/prediction timing, isolated-worker post-exec RSS, data/split/source hashes and actual data-pass/HVP/update counters, including rejected line searches.
- Sparse mini-batch and correlated-feature L-BFGS history ablations; generated Chinese/English reports, figures and retained raw measurements.
- Protocol and counter-contract tests plus the original benchmark smoke check in CI.

## 0.2.0 — Modern MVP (2026-10-14)

- C++17 `logitlab` core and `logitlab::core` CMake target with owned dense/CSR data.
- Unified averaged binary log loss, L2 regularization and unpenalized intercept across GD, mini-batch SGD, Adam, L-BFGS and Newton-CG.
- Stable sigmoid/softplus, validated public evaluators and explicit convergence/iteration/line-search outcomes.
- NumPy/SciPy Python package, GIL release and validated sparse canonicalization.
- Boost migration to pinned Abseil; fixed pinned pybind11 dependency and scikit-build-core packaging.
- Numerical/reference, validation and packaging checks; runnable C++/Python examples and expanded documentation.
- Cloud setup helper that locates its checkout and reuses retained dependencies without mandatory repeated network access.
- Historical CSV traversal/array ownership/L-BFGS accumulation repairs and opt-in legacy executables.

Kernels remain single-threaded and in-memory. Streaming/distributed training, multiclass models, sample weights and persistence are not implemented. Initial benchmark artifacts remain historical synthetic measurements.
