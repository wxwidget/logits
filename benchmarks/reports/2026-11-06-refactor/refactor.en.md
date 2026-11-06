# Runtime refactoring: quality and performance acceptance

**2× gate: PASS; geometric mean full-fit speedup = 2.0337×.**

Baseline: original 3374ad2 / 0.4.0. Candidate: refactored 0.4.1. All 18 primary workloads share identical data, objectives and full-gradient accuracy, verified independently. Four fixed-budget SGD/Adam workloads are reported separately.

Aggregation: repeat medians within each seed, then median/IQR across seeds. Speedup is the ratio of baseline/candidate medians; the primary gate is the geometric mean across all 18 ratios. Full fit includes copying, validation and initialization, rather than only the fastest native kernel segment.

## Code quality

| Audit | Before | After | CI budget |
| --- | --- | --- | --- |
| Maximum C++ function complexity | 66 | 15 | 15 |
| Maximum Python runtime function complexity | 27 | 14 | 15 |

Data contracts, storage/numeric kernels, response objectives, optimizer state and binding registration have separate responsibilities. The Python facade exports estimators; estimators depend directly on conversion/validation modules. The function gate covers the modern runtime, public headers and bindings, excluding historical experiments and generated reports. Full function inventories are retained.

![Complexity audit](complexity.png)

## Primary workloads: matched convergence accuracy

| Storage | Model/solver | Before fit ms [IQR] | After fit ms [IQR] | Speedup | Max stationarity before/after |
| --- | --- | --- | --- | --- | --- |
| csr | glm:binomial | 22.061 [21.520, 23.014] | 14.603 [14.166, 14.817] | 1.511× | 6.44e-08 / 4.73e-07 |
| dense | glm:binomial | 30.603 [30.356, 30.768] | 13.153 [12.595, 13.467] | 2.327× | 2.35e-10 / 9.41e-08 |
| csr | glm:gaussian | 23.255 [20.935, 24.476] | 11.855 [11.618, 11.897] | 1.962× | 9.89e-07 / 1.94e-07 |
| dense | glm:gaussian | 26.116 [24.847, 26.194] | 8.838 [8.808, 9.026] | 2.955× | 5.63e-09 / 4.85e-08 |
| csr | glm:poisson | 35.196 [35.116, 35.780] | 28.189 [27.846, 28.545] | 1.249× | 3.98e-07 / 4.05e-07 |
| dense | glm:poisson | 34.932 [34.771, 36.949] | 17.528 [17.274, 18.520] | 1.993× | 4.02e-07 / 4.02e-07 |
| csr | linear:cg | 22.385 [21.625, 22.567] | 10.263 [9.992, 10.704] | 2.181× | 6.67e-07 / 6.67e-07 |
| dense | linear:cg | 22.001 [21.741, 22.559] | 8.055 [8.012, 8.285] | 2.731× | 7.30e-07 / 7.30e-07 |
| csr | linear:fista | 30.302 [29.620, 30.437] | 15.106 [14.418, 15.150] | 2.006× | 8.44e-07 / 8.44e-07 |
| dense | linear:fista | 48.584 [48.029, 48.718] | 14.244 [14.056, 14.781] | 3.411× | 5.10e-07 / 5.10e-07 |
| csr | linear:proximal-gd | 23.234 [23.126, 23.319] | 11.323 [11.170, 11.732] | 2.052× | 9.66e-07 / 9.66e-07 |
| dense | linear:proximal-gd | 38.905 [38.721, 41.436] | 12.706 [12.565, 12.841] | 3.062× | 8.51e-07 / 8.51e-07 |
| csr | logistic:gd | 76.799 [76.560, 79.003] | 63.840 [63.666, 63.917] | 1.203× | 9.88e-07 / 9.88e-07 |
| dense | logistic:gd | 102.765 [100.867, 103.714] | 55.767 [54.991, 56.200] | 1.843× | 9.97e-07 / 9.97e-07 |
| csr | logistic:lbfgs | 19.586 [18.905, 19.667] | 15.650 [15.103, 16.060] | 1.251× | 6.76e-07 / 6.76e-07 |
| dense | logistic:lbfgs | 19.963 [19.702, 20.103] | 10.802 [10.526, 11.740] | 1.848× | 8.77e-07 / 8.77e-07 |
| csr | logistic:newton-cg | 26.068 [25.841, 26.873] | 14.272 [13.773, 14.362] | 1.827× | 6.71e-08 / 4.92e-07 |
| dense | logistic:newton-cg | 33.737 [33.343, 34.238] | 11.580 [11.434, 11.779] | 2.913× | 1.26e-09 / 2.17e-07 |

![Full-fit speedup](speedup.png)

## Stochastic algorithms: fixed budget

SGD/Adam share seeds, 30 epochs, batch=256 and constant learning rates. This track compares time and result consistency at fixed work budgets; budget exhaustion is not convergence and these ratios do not enter the 2× convergence gate.

| Storage | Model/solver | Before ms [IQR] | After ms [IQR] | Speedup | Before/after statuses |
| --- | --- | --- | --- | --- | --- |
| csr | logistic:adam | 129.039 [128.837, 129.746] | 122.664 [121.350, 125.638] | 1.052× | max_epochs / max_epochs |
| dense | logistic:adam | 173.328 [173.123, 174.270] | 91.871 [91.745, 91.904] | 1.887× | max_epochs / max_epochs |
| csr | logistic:sgd | 102.397 [101.649, 106.860] | 96.763 [96.633, 96.914] | 1.058× | max_epochs / max_epochs |
| dense | logistic:sgd | 177.270 [176.327, 177.564] | 93.998 [92.154, 94.016] | 1.886× | max_epochs / max_epochs |

## Core time, prediction and memory

Values below are seed medians. Prediction includes input conversion. RSS is whole-worker Linux VmHWM, including interpreter, generated data, warm-up and independent NumPy checks; it is not incremental solver memory.

| Storage | Model/solver | Core ms before/after | Predict ms before/after | RSS MiB before/after |
| --- | --- | --- | --- | --- |
| csr | glm:binomial | 17.685 / 12.298 | 1.261 / 0.842 | 108.7 / 102.8 |
| dense | glm:binomial | 26.677 / 11.356 | 1.245 / 0.877 | 113.2 / 113.3 |
| csr | glm:gaussian | 19.281 / 8.752 | 1.294 / 0.788 | 108.6 / 102.8 |
| dense | glm:gaussian | 22.226 / 7.013 | 1.181 / 0.825 | 113.2 / 113.2 |
| csr | glm:poisson | 30.437 / 24.986 | 1.278 / 0.880 | 108.6 / 102.8 |
| dense | glm:poisson | 31.182 / 15.071 | 1.141 / 0.858 | 113.2 / 113.2 |
| csr | linear:cg | 16.325 / 7.725 | 1.299 / 0.770 | 108.7 / 102.8 |
| dense | linear:cg | 18.167 / 6.169 | 1.064 / 0.812 | 113.2 / 113.2 |
| csr | linear:fista | 24.774 / 11.735 | 1.176 / 0.827 | 108.7 / 102.8 |
| dense | linear:fista | 44.797 / 12.321 | 1.163 / 0.792 | 113.2 / 113.2 |
| csr | linear:proximal-gd | 18.862 / 8.869 | 1.260 / 0.774 | 108.9 / 103.1 |
| dense | linear:proximal-gd | 35.474 / 10.790 | 1.140 / 0.762 | 113.1 / 113.3 |
| csr | logistic:adam | 124.344 / 119.106 | 1.239 / 0.821 | 108.7 / 102.9 |
| dense | logistic:adam | 170.025 / 89.523 | 1.248 / 0.774 | 113.2 / 113.1 |
| csr | logistic:gd | 71.817 / 60.720 | 1.357 / 0.907 | 108.7 / 102.9 |
| dense | logistic:gd | 95.374 / 53.565 | 1.322 / 0.791 | 113.2 / 113.2 |
| csr | logistic:lbfgs | 15.186 / 12.792 | 1.347 / 0.877 | 108.6 / 102.8 |
| dense | logistic:lbfgs | 16.036 / 8.200 | 1.294 / 0.774 | 113.2 / 113.3 |
| csr | logistic:newton-cg | 20.703 / 11.529 | 1.333 / 0.893 | 108.6 / 102.8 |
| dense | logistic:newton-cg | 29.747 / 9.308 | 1.190 / 0.788 | 113.2 / 113.2 |
| csr | logistic:sgd | 97.120 / 92.353 | 1.335 / 0.823 | 108.7 / 102.8 |
| dense | logistic:sgd | 171.945 / 88.986 | 1.297 / 0.776 | 113.2 / 113.3 |

## Held-out quality

Seed medians below use Binomial log loss or Gaussian/Poisson MSE. Every paired repeat is checked independently; aggregation cannot hide a quality failure.

| Storage | Model/solver | Quality before | Quality after |
| --- | --- | --- | --- |
| csr | glm:binomial | 0.67989918 | 0.67989919 |
| dense | glm:binomial | 0.65031567 | 0.65031567 |
| csr | glm:gaussian | 0.27479837 | 0.27479837 |
| dense | glm:gaussian | 0.25653845 | 0.25653845 |
| csr | glm:poisson | 1.78795622 | 1.78795620 |
| dense | glm:poisson | 1.56489686 | 1.56489689 |
| csr | linear:cg | 0.27479835 | 0.27479835 |
| dense | linear:cg | 0.25653843 | 0.25653843 |
| csr | linear:fista | 0.40274933 | 0.40274933 |
| dense | linear:fista | 0.27374376 | 0.27374376 |
| csr | linear:proximal-gd | 0.40274958 | 0.40274958 |
| dense | linear:proximal-gd | 0.27374386 | 0.27374386 |
| csr | logistic:adam | 0.68853396 | 0.68853396 |
| dense | logistic:adam | 0.65079070 | 0.65079070 |
| csr | logistic:gd | 0.67989909 | 0.67989909 |
| dense | logistic:gd | 0.65031569 | 0.65031569 |
| csr | logistic:lbfgs | 0.67989921 | 0.67989921 |
| dense | logistic:lbfgs | 0.65031559 | 0.65031559 |
| csr | logistic:newton-cg | 0.67989918 | 0.67989917 |
| dense | logistic:newton-cg | 0.65031567 | 0.65031569 |
| csr | logistic:sgd | 0.68078857 | 0.68078857 |
| dense | logistic:sgd | 0.65022242 | 0.65022242 |

## Protocol, provenance and limits

- CPU: INTEL(R) XEON(R) PLATINUM 8573C; affinity: [0]; threads: {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}.
- Python 3.12.14; NumPy 2.3.5; SciPy 1.17.0; libc ['glibc', '2.41'].
- Measured: 2026-11-06T21:13:46.561671+08:00; seeds: [42, 43, 44]; repeats: 3.
- Dense: 24,000 × 96; CSR: 24,000 × 2,000, density 1%; training 80%, held-out 20%.
- L2=.1; linear proximal solvers additionally use L1=.01. Unpenalized intercept; primary tolerance=1e-6.
- SGD learning rate=.03; Adam=.003. Primary budgets: logistic/GLM 300, linear 1,000; actual work retained.
- Baseline runtime: {'simd': 'compiler_default'}; candidate: {'simd': 'avx2_fma', 'vector_math': 'glibc_libmvec'}.

Every primary fit must both report convergence and satisfy an independently recomputed gradient/proximal mapping norm ≤1.00001e−6. Paired objectives agree within 1e−9, coefficients within 3e−5 absolute plus 1e−5 relative tolerance, and held-out metrics within 2e−5. Quality is Binomial log loss or Gaussian/Poisson MSE. Optimization does not relax these thresholds.

Native binaries, sources, datasets and actual build configurations are hashed. Both artifacts use the same Release compiler and base optimization level. The candidate runtime-detects optional SIMD/system vector math without fast-math or extra threads. Some solvers avoid scans or unnecessary inner work, so this is not an equal-instruction microbenchmark.

These measurements use fixed-size synthetic data on the recorded CPU. CPU frequency and host contention remain uncontrolled despite core/thread pinning; three seeds do not cover every data condition. A geometric-mean pass does not imply every workload passes. Remeasure when optional backends are unavailable. Scalar/vector math can differ in final-bit rounding.

[Raw measurements and histories](../../results/2026-11-06-refactor/results.json) · [Baseline audit](../../results/2026-11-06-refactor/quality-before.json) · [Candidate audit](../../results/2026-11-06-refactor/quality-after.json)

Design, exact commands and ownership/numerical contracts: [refactoring guide](../../../docs/refactoring.md). Content hashes for this report and its generator are in `report-manifest.json`.

[Development experiments, including failed gates](../../results/2026-11-06-refactor/development-attempts.md)
