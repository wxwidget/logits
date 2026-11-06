# 代码质量重构与性能验收

**2× gate: 通过; geometric mean full-fit speedup = 2.0646×.**

基线为原始 3374ad2 / 0.4.0，候选为重构后的 0.4.1。所有 18 个主场景均使用相同数据、目标函数和完整梯度精度，且已通过独立数值检查。SGD/Adam 的 4 个固定预算场景单独报告。

汇总规则：先取每个种子内重复中位数，再跨种子取中位数及 IQR。加速比是基线/候选中位数之比，主指标为 18 项加速比的几何平均。完整 fit 包括输入复制、校验和初始化；不是只计最快的 C++ 内核片段。

## 代码质量

| Audit | Before | After | CI budget |
| --- | --- | --- | --- |
| Maximum C++ function complexity | 66 | 15 | 15 |
| Maximum Python runtime function complexity | 27 | 14 | 15 |

数据契约、存储/数值内核、模型目标、求解器状态和绑定注册分别负责独立职责；Python 公开模块只负责导出，估计器直接依赖转换/校验模块。审查范围是现代运行时、公开头文件和绑定，历史实验与生成报告不计入该函数门禁。完整函数清单随原始数据保留。

![Complexity audit](complexity.png)

## 主场景：收敛到相同精度

| Storage | Model/solver | Before fit ms [IQR] | After fit ms [IQR] | Speedup | Max stationarity before/after |
| --- | --- | --- | --- | --- | --- |
| csr | glm:binomial | 23.053 [22.845, 23.224] | 14.863 [14.821, 15.380] | 1.551× | 6.44e-08 / 4.73e-07 |
| dense | glm:binomial | 31.425 [30.059, 32.923] | 12.486 [12.412, 12.692] | 2.517× | 2.35e-10 / 9.41e-08 |
| csr | glm:gaussian | 25.933 [23.265, 26.590] | 11.680 [11.635, 12.335] | 2.220× | 9.89e-07 / 1.94e-07 |
| dense | glm:gaussian | 36.571 [30.420, 43.164] | 9.458 [8.951, 9.469] | 3.867× | 5.63e-09 / 4.85e-08 |
| csr | glm:poisson | 35.528 [34.768, 36.079] | 29.371 [28.533, 29.996] | 1.210× | 3.98e-07 / 4.05e-07 |
| dense | glm:poisson | 35.072 [34.186, 38.486] | 18.081 [17.574, 19.341] | 1.940× | 4.02e-07 / 4.02e-07 |
| csr | linear:cg | 21.303 [20.351, 21.793] | 11.391 [10.941, 11.572] | 1.870× | 6.67e-07 / 6.67e-07 |
| dense | linear:cg | 22.942 [22.861, 23.033] | 8.125 [8.028, 8.471] | 2.824× | 7.30e-07 / 7.30e-07 |
| csr | linear:fista | 29.713 [28.884, 29.950] | 15.305 [15.302, 15.990] | 1.941× | 8.44e-07 / 8.44e-07 |
| dense | linear:fista | 48.328 [47.680, 53.975] | 14.486 [14.155, 14.929] | 3.336× | 5.10e-07 / 5.10e-07 |
| csr | linear:proximal-gd | 23.710 [23.221, 23.999] | 11.931 [11.844, 11.959] | 1.987× | 9.66e-07 / 9.66e-07 |
| dense | linear:proximal-gd | 40.765 [40.358, 40.828] | 11.908 [11.840, 12.304] | 3.423× | 8.51e-07 / 8.51e-07 |
| csr | logistic:gd | 78.518 [78.193, 79.458] | 64.698 [64.523, 65.205] | 1.214× | 9.88e-07 / 9.88e-07 |
| dense | logistic:gd | 98.539 [98.440, 99.202] | 53.744 [53.022, 54.866] | 1.834× | 9.97e-07 / 9.97e-07 |
| csr | logistic:lbfgs | 20.392 [19.743, 20.814] | 16.000 [15.702, 16.395] | 1.275× | 6.76e-07 / 6.76e-07 |
| dense | logistic:lbfgs | 18.373 [18.316, 18.663] | 10.546 [10.188, 10.660] | 1.742× | 8.77e-07 / 8.77e-07 |
| csr | logistic:newton-cg | 25.912 [25.542, 26.495] | 15.152 [14.808, 15.243] | 1.710× | 6.71e-08 / 4.92e-07 |
| dense | logistic:newton-cg | 34.339 [33.541, 35.532] | 11.429 [11.402, 11.752] | 3.004× | 1.26e-09 / 2.17e-07 |

![Full-fit speedup](speedup.png)

## 随机算法：固定预算

SGD/Adam 使用相同种子、30 个 epoch、batch=256 和固定学习率。此处比较同预算的实际时间及结果一致性，不把预算耗尽称为收敛，也不计入 2× 收敛验收指标。

| Storage | Model/solver | Before ms [IQR] | After ms [IQR] | Speedup | Before/after statuses |
| --- | --- | --- | --- | --- | --- |
| csr | logistic:adam | 133.102 [130.203, 135.557] | 123.656 [121.386, 124.229] | 1.076× | max_epochs / max_epochs |
| dense | logistic:adam | 186.978 [185.636, 187.875] | 102.111 [100.122, 102.544] | 1.831× | max_epochs / max_epochs |
| csr | logistic:sgd | 103.966 [103.532, 104.074] | 100.898 [99.035, 101.900] | 1.030× | max_epochs / max_epochs |
| dense | logistic:sgd | 177.711 [177.626, 178.974] | 94.218 [92.066, 94.223] | 1.886× | max_epochs / max_epochs |

## 核心、预测与内存

以下均为种子中位数。预测耗时含输入转换；RSS 为整个 worker 的 Linux VmHWM，包含解释器、数据生成、预热、NumPy 独立验算和数据，不能解释为求解器增量内存。

| Storage | Model/solver | Core ms before/after | Predict ms before/after | RSS MiB before/after |
| --- | --- | --- | --- | --- |
| csr | glm:binomial | 18.404 / 12.185 | 1.295 / 0.943 | 108.7 / 102.9 |
| dense | glm:binomial | 25.991 / 10.760 | 1.278 / 0.849 | 113.2 / 113.2 |
| csr | glm:gaussian | 21.093 / 9.016 | 1.241 / 0.926 | 108.6 / 102.8 |
| dense | glm:gaussian | 32.305 / 7.425 | 1.307 / 0.832 | 113.2 / 113.2 |
| csr | glm:poisson | 30.678 / 26.205 | 1.243 / 1.004 | 108.6 / 102.8 |
| dense | glm:poisson | 30.440 / 15.514 | 1.179 / 0.873 | 113.1 / 113.2 |
| csr | linear:cg | 15.831 / 8.168 | 1.295 / 0.965 | 108.7 / 102.8 |
| dense | linear:cg | 19.123 / 6.349 | 1.173 / 0.760 | 113.2 / 113.3 |
| csr | linear:fista | 24.902 / 12.303 | 1.164 / 0.981 | 108.7 / 102.8 |
| dense | linear:fista | 43.476 / 12.411 | 1.125 / 0.800 | 113.1 / 113.3 |
| csr | linear:proximal-gd | 18.355 / 9.094 | 1.268 / 0.983 | 108.9 / 103.1 |
| dense | linear:proximal-gd | 36.797 / 9.876 | 1.104 / 0.777 | 113.2 / 113.3 |
| csr | logistic:adam | 126.975 / 118.664 | 1.267 / 0.999 | 108.6 / 102.9 |
| dense | logistic:adam | 180.091 / 93.161 | 1.267 / 0.813 | 113.2 / 113.3 |
| csr | logistic:gd | 72.617 / 60.259 | 1.285 / 1.010 | 108.7 / 102.9 |
| dense | logistic:gd | 92.530 / 49.599 | 1.238 / 0.737 | 113.1 / 113.2 |
| csr | logistic:lbfgs | 14.479 / 12.857 | 1.326 / 0.967 | 108.6 / 102.8 |
| dense | logistic:lbfgs | 15.119 / 8.282 | 1.240 / 0.733 | 113.2 / 113.1 |
| csr | logistic:newton-cg | 21.371 / 11.246 | 1.299 / 0.993 | 108.6 / 102.8 |
| dense | logistic:newton-cg | 30.992 / 9.420 | 1.193 / 0.742 | 113.2 / 113.2 |
| csr | logistic:sgd | 97.012 / 97.223 | 1.342 / 0.978 | 108.7 / 102.8 |
| dense | logistic:sgd | 174.216 / 90.874 | 1.253 / 0.743 | 113.2 / 113.2 |

## 测试集效果

以下为各种子中位数；二项使用 log loss，高斯/泊松使用 MSE。每一对重复的效果差均单独检查，不能通过平均抵消不合格结果。

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

## 协议、来源与边界

- CPU: INTEL(R) XEON(R) PLATINUM 8573C; affinity: [0]; threads: {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}.
- Python 3.12.14; NumPy 2.3.5; SciPy 1.17.0; libc ['glibc', '2.41'].
- Measured: 2026-11-06T20:46:31.211664+08:00; seeds: [42, 43, 44]; repeats: 3.
- Dense: 24,000 × 96; CSR: 24,000 × 2,000, density 1%; training 80%, held-out 20%.
- L2=.1; linear proximal solvers additionally use L1=.01. Unpenalized intercept; primary tolerance=1e-6.
- SGD learning rate=.03; Adam=.003. Primary budgets: logistic/GLM 300, linear 1,000; actual work retained.
- Baseline runtime: {'simd': 'compiler_default'}; candidate: {'simd': 'avx2_fma', 'vector_math': 'glibc_libmvec'}.

每次主拟合必须同时满足收敛状态和独立重算的梯度/近端映射范数 ≤1.00001e−6。配对目标差 ≤1e−9、系数差 ≤3e−5（另有1e−5相对项）、测试指标差 ≤2e−5。指标为二项 log loss，或高斯/泊松 MSE；这些阈值没有因优化而放宽。

二进制、源文件和数据校验和、实际编译配置均保留。两边使用相同 Release 编译器和基础优化级别，候选通过运行时检测使用可选 SIMD/系统向量数学，未启用 fast-math 或额外线程。部分求解器减少了重复扫描/内层工作，因此不是相同指令数的微基准。

结果来自固定规模的合成数据和当前 CPU。虽固定了单核与线程数，仍未隔离 CPU 频率和宿主机负载；三个种子不能覆盖所有数据条件。几何平均达标不代表每个场景都达标；可选后端不可用时应重新测量。标量与向量数学可能有末位舍入差异。

[Raw measurements and histories](../../results/2026-11-06-refactor/results.json) · [Baseline audit](../../results/2026-11-06-refactor/quality-before.json) · [Candidate audit](../../results/2026-11-06-refactor/quality-after.json)

Design, exact commands and ownership/numerical contracts: [refactoring guide](../../../docs/refactoring.md). Content hashes for this report and its generator are in `report-manifest.json`.

[包含未达标版本的开发实验记录](../../results/2026-11-06-refactor/development-attempts.md)
