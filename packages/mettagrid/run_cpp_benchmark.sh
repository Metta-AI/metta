#!/bin/bash
set -euo pipefail

CONFIG="${1:-release}"
shift || true

cd "$(dirname "$0")"

# Discover Python runtime locations for embedded interpreter use, and point
# Bazel's Python toolchain at the same interpreter. This keeps the
# pybind11-embedded interpreter and the runtime in sync.
PY_BIN=$(command -v python3 || command -v python)
UV_PY_PREFIX="$($PY_BIN -c 'import sys; print(sys.base_prefix)')"
UV_PY_LIBDIR="$($PY_BIN -c 'import sysconfig;print(sysconfig.get_config_var("LIBDIR") or sysconfig.get_config_var("LIBPL"))')"
UV_SITEPKG="$($PY_BIN -c 'import site; print(site.getsitepackages()[0])')"

export PYTHON_BIN_PATH="$PY_BIN"
export PYTHON_LIB_PATH="$UV_PY_LIBDIR"

echo "==> Building C++ benchmark with --config=${CONFIG}..."
bazel clean
bazel build --config="${CONFIG}" //benchmarks:test_mettagrid_env_benchmark

echo "==> Running C++ benchmark binary..."
# Run binary with correct Python runtime env so libpython and stdlib/site-packages
# are discoverable from the embedded interpreter.
LD_LIBRARY_PATH="${UV_PY_LIBDIR}:${LD_LIBRARY_PATH:-}" \
  PYTHONHOME="${UV_PY_PREFIX}" \
  PYTHONPATH="${UV_SITEPKG}:${UV_PY_PREFIX}/lib/python3.12:${UV_PY_PREFIX}/lib/python3.12/lib-dynload" \
  ./bazel-bin/benchmarks/test_mettagrid_env_benchmark \
  --benchmark_time_unit=us \
  --benchmark_min_warmup_time=5 \
  --benchmark_repetitions=1 \
  --benchmark_min_time=5s "$@"
