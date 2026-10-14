#!/usr/bin/env bash
# Reusable local/cloud setup. Generated files stay in an ignored or external directory.
set -euo pipefail
logitlab_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
logitlab_venv=${LOGITLAB_VENV:-"$logitlab_repo/.venv"}
logitlab_build=${LOGITLAB_BUILD_DIR:-"$logitlab_repo/build/cloud"}
python -m venv "$logitlab_venv"
export PATH="$logitlab_venv/bin:$PATH"
if [[ -z "${CXX:-}" ]] || ! command -v "$CXX" >/dev/null; then
  export CXX=$(command -v g++)
fi
export CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-2}
if ! python -c 'import numpy, scipy, pytest, pytest_cov, build, scikit_build_core' 2>/dev/null \
    || [[ ! -x "$logitlab_venv/bin/cmake" || ! -x "$logitlab_venv/bin/ninja" ]]; then
  python -m pip install --disable-pip-version-check 'cmake>=3.24' ninja \
    'scikit-build-core>=0.10,<2' 'numpy>=1.23' 'scipy>=1.9' 'pytest>=7' \
    'pytest-cov>=5' 'build>=1.2,<2'
fi
cd "$logitlab_repo"
cmake -S . -B "$logitlab_build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$CXX" -DPython_EXECUTABLE="$logitlab_venv/bin/python" \
  -DLOGITLAB_PYTHON=ON -DLOGITLAB_LEGACY=OFF -DBUILD_TESTING=ON \
  -DFETCHCONTENT_FULLY_DISCONNECTED=OFF -DFETCHCONTENT_UPDATES_DISCONNECTED=ON
cmake --build "$logitlab_build" --parallel "$CMAKE_BUILD_PARALLEL_LEVEL"
ctest --test-dir "$logitlab_build" --output-on-failure
# Reuse verified dependency checkouts so wheel builds do not contact GitHub again.
export CMAKE_ARGS="${CMAKE_ARGS:-} -DFETCHCONTENT_SOURCE_DIR_ABSEIL=\"$logitlab_build/_deps/abseil-src\" -DFETCHCONTENT_SOURCE_DIR_PYBIND11=\"$logitlab_build/_deps/pybind11-src\""
python -m pip install --disable-pip-version-check --no-build-isolation --no-deps .
python -m pip check
python -m pytest -q --cov=logitlab --cov-report=term-missing --cov-fail-under=95
python examples/train.py
