# Development and cloud recovery

## Standard build

```bash
python -m pip install cmake ninja
cmake --preset release
cmake --build --preset release --parallel 2
ctest --preset release
python -m pip install '.[test]'
python -m pytest -q --cov=logitlab --cov-report=term-missing --cov-fail-under=95
```

CMake uses fixed Git commits for Abseil 20250127.1 and pybind11 2.13.6. FetchContent manages C++ source dependencies; scikit-build-core builds the Python wheel. Python dependency ranges are not a fully locked environment. Boost is not required. The Python coverage report measures wrapper code, not C++ kernels; C++ numerical tests and sanitizers cover the core separately.

| CMake option | Default | Effect |
| --- | --- | --- |
| `BUILD_TESTING` | `ON` in standalone builds | Build C++ numerical tests |
| `LOGITLAB_EXAMPLES` | `ON` | Build the runnable C++ example |
| `LOGITLAB_PYTHON` | `OFF` standalone, `ON` for wheels | Build `_core` with pybind11 |
| `LOGITLAB_LEGACY` | `OFF` | Historical executables and bundled static archive |
| `LOGITLAB_NATIVE` | `OFF` | GCC/Clang `-march=native`; binaries are CPU-specific |

Portable Release uses normal compiler optimization, without fast-math. To select a compiler set `CXX` before the first configure or use `-DCMAKE_CXX_COMPILER`. Use a separate build directory when changing toolchains.

## Reusable environment helper

```bash
bash tools/setup_environment.sh
```

The helper locates the checkout from its own script path, creates/reuses `.venv` and `build/cloud`, checks development dependencies, configures/builds the C++ core and Python extension, installs the wheel from local sources, checks dependencies, runs CTest/Python tests and executes the Python example. It does not rely on the repository folder being named `regressionlib`, `logits` or `logitlab`.

Cloud tasks can retain outputs outside the checkout:

```bash
LOGITLAB_VENV=/workspace/.logitlab-venv \
LOGITLAB_BUILD_DIR=/workspace/.logitlab-build \
bash tools/setup_environment.sh
```

The helper reuses installed packages and verified FetchContent source caches. Initial setup can need network access; repeat setup with unchanged pins can run without network. A changed pin absent from the cache requires a normal FetchContent update with `FETCHCONTENT_UPDATES_DISCONNECTED=OFF`. It never disables TLS/signature verification. The cloud Git HTTPS proxy supplies repository authentication; GitHub API tokens are not needed for building or for Git operations supported by that proxy.

A saved environment draft is separate from the running instance. If the terminal is unavailable while the environment is `starting`, the helper cannot diagnose machine provisioning. Once connected, run it and inspect the failing command rather than assuming a token or folder rename is the cause. Network operations may need the execution platform's allowed network context.

## Offline wheel rebuild

After the helper has prepared dependency sources:

```bash
export CMAKE_ARGS="-DFETCHCONTENT_SOURCE_DIR_ABSEIL=$PWD/build/cloud/_deps/abseil-src -DFETCHCONTENT_SOURCE_DIR_PYBIND11=$PWD/build/cloud/_deps/pybind11-src"
python -m pip install --no-build-isolation --no-deps --disable-pip-version-check .
```

The activated environment must already contain runtime and build dependencies. Source overrides must match the commits in `CMakeLists.txt`; supplying arbitrary versions does not verify dependency integrity.

## Sanitizers

```bash
cmake -S . -B build/asan -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer'
cmake --build build/asan --parallel 2
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 \
  ctest --test-dir build/asan --output-on-failure
```

Run standalone C++ tests/examples under sanitizers. Historical executables are not part of modern solver equivalence checks.

## Source distribution and wheel

```bash
python -m build --sdist
python -m pip wheel --no-deps dist/logitlab-0.2.0.tar.gz -w dist
python -m pip install --force-reinstall --no-deps dist/logitlab-0.2.0-*.whl
python -m pytest -q
```

First builds fetch dependencies. In a prepared environment you can use `--no-isolation`/`--no-build-isolation` and the pinned source overrides above. Test the installed wheel from outside the checkout as well as running the suite; importing source files alone does not validate packaging.

## Historical experiments

```bash
cmake -S . -B build/legacy -DCMAKE_BUILD_TYPE=Release -DLOGITLAB_LEGACY=ON
cmake --build build/legacy --parallel 2
./build/legacy/legacy_lr data/train.csv data/trainLabels.csv 1000 40
```

The historical LR/FTRL/L-BFGS programs use Abseil and retain their original conventions. They do not share the modern research objective. The bundled `lib/liblbfgs.a` is platform-specific and needs non-PIE linking on the verified Linux x86_64 setup. Old Python2/pytron experiments are archived and unsupported by this workflow.
