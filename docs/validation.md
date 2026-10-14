# Validation record — 2026-10-10

Local validation: GCC 14.2.0, C++17, CMake 4.4.4, portable Release (`-O3 -DNDEBUG`), Linux x86_64, Python 3.12.14. Kernels are single-threaded; this shared cloud machine had five available CPUs. No throughput ranking across machines is implied.

13 Python tests passed. C++ checks passed in Release and in AddressSanitizer/UBSan Debug builds. All seven opt-in historical executables compiled; legacy LR, L-BFGS and FTRL training/prediction completed. A Python wheel and source distribution were built. GitHub CI is configured but has not been run remotely.

Synthetic smoke benchmarks use three seeds (42, 43, 44), an 80/20 split, training-only RMS scaling, L2=0.01, batch size 256 and full-gradient tolerance 1e-6. The dense experiment uses 10,000 × 100 input and 30 outer iterations/epochs; the sparse experiment uses 100,000 × 10,000 input at density 0.0005 and 10 outer iterations/epochs. Times include Python conversion and are single-run measurements per seed, not stable performance estimates.

| Experiment | Solver | Median fit seconds | Median train objective | Median validation log loss | Converged seeds |
| --- | --- | ---: | ---: | ---: | ---: |
| dense | adam | 0.1418 | 0.617907 | 0.629412 | 0/3 |
| dense | gd | 0.0482 | 0.641999 | 0.647164 | 0/3 |
| dense | lbfgs | 0.0126 | 0.591171 | 0.606431 | 3/3 |
| dense | newton-cg | 0.0253 | 0.591171 | 0.606431 | 3/3 |
| dense | sgd | 0.1527 | 0.591252 | 0.606460 | 0/3 |
| sparse | adam | 0.4122 | 7.236350 | 5.856981 | 0/3 |
| sparse | gd | 0.0832 | 0.664139 | 0.677469 | 0/3 |
| sparse | lbfgs | 0.0908 | 0.551123 | 0.687636 | 0/3 |
| sparse | newton-cg | 0.2041 | 0.551123 | 0.687653 | 3/3 |
| sparse | sgd | 0.1836 | 0.614569 | 0.765278 | 0/3 |
| sparse-adam-tuned | adam | 0.3592 | 0.574635 | 0.716361 | 0/3 |

Default Adam step size 0.05 performs poorly on these standardized sparse features. A separate run at 0.001 improves it but still does not reach the gradient tolerance in ten epochs. This observation is a hyperparameter sensitivity result, not evidence that one solver is universally superior. SGD also requires tuning. The original poor results remain in the raw artifacts.

Raw JSON and per-epoch traces are in [benchmarks/results/2026-10-10](../benchmarks/results/2026-10-10/). The initial dense/sparse JSON predates the CLI hyperparameter flags; those two runs used fixed learning_rate=0.05, l2=0.01, batch_size=256, tolerance=1e-6. The tuned JSON explicitly records its options. Current benchmark options expose these parameters.

Reproduction commands:

```bash
python benchmarks/compare.py --rows 10000 --cols 100 --epochs 30
python benchmarks/compare.py --rows 100000 --cols 10000 --density 0.0005 --epochs 10
python benchmarks/compare.py --rows 100000 --cols 10000 --density 0.0005 --epochs 10 --learning-rate .001 --solvers adam
```
