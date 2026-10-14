# Getting started

Follow the repository [installation steps](../README.md#quick-start), then run:

```bash
python examples/train.py
```

The example fits the same four-row classification problem from a NumPy array and a SciPy CSR matrix. Both should print `converged` and predicted labels `[0 0 1 1]`. This is an API demonstration, not a held-out accuracy benchmark.

## Compare solvers on a train/validation split

```bash
python benchmarks/compare.py --rows 10000 --cols 100 --epochs 30 \
  --seeds 42 43 44 --output build/dense-experiment
```

Open `results.json` for per-seed validation metrics and `traces.csv` for optimization trajectories. Compare loss and gradient norm at similar elapsed times. A `max_epochs` status means the tolerance was not reached. Increasing the budget, scaling features, or tuning step/batch size can be necessary. Use a separate tuning protocol before judging solver performance.

## Sparse data

The Python API accepts CSR, CSC, COO and other SciPy sparse inputs. It validates compressed formats before canonicalization, owns a CSR copy and never allocates a dense `n × p` feature matrix. Scaling should preserve zeros: the benchmark uses training-only RMS scale without mean subtraction.

```bash
python benchmarks/compare.py --rows 100000 --cols 10000 --density .0005 \
  --epochs 10 --learning-rate .001 --solvers adam lbfgs \
  --output build/sparse-experiment
```

These data fit in memory; the command does not implement streaming. Sparse mini-batches still update dense model/moment vectors, so a very large feature count can dominate runtime.

## C++

```bash
cmake --preset release
cmake --build --preset release --parallel 2
./build/release/logitlab_train
ctest --preset release
```

The complete [example](../examples/train.cpp) constructs a `logitlab::Dataset`, trains with `logitlab::fit` and produces positive-class probabilities with `logitlab::predict`. The example is registered with CTest and returns nonzero if it does not converge. Use C++17 and link `logitlab::core` from a CMake build containing this project.

## Troubleshooting

- **Compiler missing:** install a C++17 compiler. In the cloud setup helper, an unavailable `CXX` is replaced with installed `g++`.
- **First dependency fetch fails:** check HTTPS access to GitHub and PyPI. Fixed dependency source directories can be supplied for offline builds; see [development](development.md).
- **`max_epochs`:** inspect the gradient norm and scale features; this status is an optimization result, not a successful convergence claim.
- **Invalid CSR/labels:** check matrix structure and map binary labels explicitly to 0/1 before training.
- **Cloud terminal unavailable:** a saved configuration draft does not start a machine. Inspect the environment's startup status; once connected, the setup helper checks the actual filesystem and build.
