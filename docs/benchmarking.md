# Benchmark protocol

## Kaggle real-data study

[English protocol and commands](kaggle-benchmark.md) · [中文说明](kaggle-benchmark.zh-CN.md)

The Kaggle runner covers correlated numeric classification (Breast Cancer), imbalanced one-hot CSR classification (Adult), and mixed-feature linear regression (Insurance). It downloads versioned public archives with observed-license and checksum records, learns preprocessing only on training rows, compares raw/RMS, and generates matching English/Chinese quality and performance reports.

**Current status:** offline protocol tests pass; actual downloads/measurements await activation of the cloud network allowlist. No real Kaggle results are claimed. The linked guide explains the exact environment-setting step and the independent checks completed.

## Linear regression, model selection and constraints

- [English regression report](../benchmarks/reports/2026-10-16-linear/linear.en.md)
- [线性回归中文报告](../benchmarks/reports/2026-10-16-linear/linear.zh-CN.md)

```bash
python -m pip install '.[test,benchmark]'
python benchmarks/linear_suite.py --output build/linear-benchmark
python benchmarks/linear_report.py --input build/linear-benchmark --output build/linear-report
```

The linear suite compares OLS/Ridge/Lasso/Elastic Net, WLS under known heteroscedasticity, and nonnegative/box-fixed domain priors. Six synthetic cases use fixed shared train/validation/test splits, three seeds and three measured repetitions. Ridge/Lasso/Elastic Net select α from four candidates using independent reference fits and validation MSE, then every applicable C++ solver receives the same selected objective. Test labels never select parameters.

NumPy/SciPy solve independently formulated weighted/augmented least squares and bounds; scikit-learn coordinate descent verifies L1. These are optional benchmark dependencies, not C++ runtime dependencies. Raw/RMS feature representations preserve OLS predictions; regularized pipelines reselect penalties because scaling changes their meaning. Bounds and fixed values are transformed into the correct input coordinates.

Reports distinguish reference predictive quality, actual finite-budget quality, stationarity, normalized objective gaps, final/ever target counts, data passes, latency, prediction throughput and whole-worker post-exec peak memory. Full iteration traces retain repeat zero; all timing repetitions retain metrics and first-target evidence. Reports reject missing/duplicate measurement slots.

## Measured optimizer study

- [English report](../benchmarks/reports/2026-10-15/benchmark.en.md)
- [中文报告](../benchmarks/reports/2026-10-15/benchmark.zh-CN.md)
- [Raw metrics, references and calibration trials](../benchmarks/results/2026-10-15/results.json)

```bash
python -m pip install '.[test,benchmark]'
python benchmarks/suite.py --output build/benchmark-suite
python benchmarks/report.py --input build/benchmark-suite --output build/benchmark-report
```

The suite uses five scenarios and stratified train/validation/test splits, raw versus training-only RMS, three seeds and three fit repetitions. Its λ=0 track isolates coordinate conditioning; λ=0.01 examines the changed regularization/pipeline. It preserves canonical CSR and uses no test labels to select parameters. Synthetic multiscale scenarios change units after generating labels, so their signal does not accidentally change with scale.

Step-size grids minimize final training objective, with all trials retained. Three trials are used for GD/SGD/Adam, one for L-BFGS/Newton-CG. Calibration budgets differ; timing tables exclude calibration from fit and report it separately. This is a bounded optimization study, not equivalent hyperparameter tuning or validation-based model selection. Batch-size and history-length ablations characterize stochastic update cost and curvature history.

The reference independently implements mean log loss/gradient and exact HVP in SciPy, with objective-preserving penalty transformation under coordinate preconditioning. Invalid references suppress objective-gap and target statistics. Target time is the first recorded C++ step reaching a 1e-4 gap, conditional on reaching; final-target counts separately identify later drift. Equal epochs do not mean equal work. Core counters include backtracking, monitoring, curvature, HVP and mini-batch visits, with dense parameter updates recorded separately.

JSON includes source/binary and dataset/split/scaler hashes, CPU affinity, compiler availability and NumPy/BLAS metadata. Reports aggregate repetitions within seeds before aggregating across seeds. RSS is whole-worker peak, not solver allocation. Prediction uses seven warmed API calls and includes input copying. Inputs/outputs are in-memory, kernels are single-threaded, and CPU frequency/background load are not isolated. Larger workloads can be studied with the configurable smoke runner or by extending the suite scenario generator.

On Linux, worker peak memory uses the post-exec address space's `/proc/self/status` VmHWM; `getrusage().ru_maxrss` can inherit a parent's pre-exec peak and contaminate comparisons. A regression test launches a worker from a parent with a large allocation to check this distinction. For actual build flags, supply `--build-cache /path/to/installed-wheel-build`; the measured study retains the installed wheel's CMake configuration and build commands in JSON.

## General research checklist

1. Fix dataset generation, train/validation split, preprocessing and objective normalization. Use multiple seeds. For real datasets, document provenance and checksums, label mapping, missing-value treatment and feature scaling.
2. Run a portable Release build first. Record compiler version, build flags, CPU model, available RAM, dependency revisions and thread configuration beside results. The runner records runtime metadata, but compiler and hardware details must be added by the experiment owner.
3. Check gradients and compare solutions with a trusted independently formulated objective. Distinguish a tolerance stop, epoch exhaustion and a failed line search.
4. Tune solver hyperparameters on a separate validation protocol with equivalent search budgets. Defaults are baselines, not equally tuned configurations.
5. Compare convergence traces versus elapsed time, not just iterations. Report full-gradient norm, penalized training objective and unpenalized held-out log loss. Validation accuracy alone hides calibration and optimization differences.
6. Separate Python conversion costs, C++ optimization and prediction. Add warm-up and repeated runs for throughput claims; control CPU frequency, background load and memory pressure. Measure peak RSS externally when investigating memory. A single synthetic run is only a smoke benchmark.

The runner saves raw per-seed metrics and traces; it intentionally does not invent a winner or a speedup baseline. SGD/Adam may oscillate with constant step sizes. Large sparse feature spaces currently incur dense O(p) work per mini-batch, so feature count is as important as nonzero count.
