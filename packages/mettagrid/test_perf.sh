#!/bin/bash
set -euo pipefail

# test_perf.sh - Build and run performance test for MettaGrid
# This script:
# 1. Optionally configures Bazel via METTAGRID_BAZEL_CONFIG / --bazel-config
# 2. Uses uv to rebuild the package (builds .so via bazel_build.py)
# 3. Runs the Python performance test

# Allow selecting Bazel config and output root via CLI flags so we can
# compare different compilers/optimization levels reproducibly.
BAZEL_CONFIG="${METTAGRID_BAZEL_CONFIG:-}"
BAZEL_OUTPUT_ROOT="${METTAGRID_BAZEL_OUTPUT_ROOT:-}"
PY_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bazel-config)
      if [[ $# -lt 2 ]]; then
        echo "--bazel-config requires a value" >&2
        exit 1
      fi
      BAZEL_CONFIG="$2"
      shift 2
      ;;
    --bazel-config=*)
      BAZEL_CONFIG="${1#*=}"
      shift
      ;;
    --bazel-output-root)
      if [[ $# -lt 2 ]]; then
        echo "--bazel-output-root requires a value" >&2
        exit 1
      fi
      BAZEL_OUTPUT_ROOT="$2"
      shift 2
      ;;
    --bazel-output-root=*)
      BAZEL_OUTPUT_ROOT="${1#*=}"
      shift
      ;;
    --)
      shift
      PY_ARGS+=("$@")
      break
      ;;
    *)
      PY_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -n "$BAZEL_CONFIG" ]]; then
  export METTAGRID_BAZEL_CONFIG="$BAZEL_CONFIG"
fi

if [[ -n "$BAZEL_OUTPUT_ROOT" ]]; then
  export METTAGRID_BAZEL_OUTPUT_ROOT="$BAZEL_OUTPUT_ROOT"
fi

cd "$(dirname "$0")"

# echo "==> Cleaning Bazel build artifacts..."
# bazel clean --expunge

echo "==> Rebuilding mettagrid package (builds .so via bazel_build.py)..."
cd ../..
uv sync --reinstall-package mettagrid
cd packages/mettagrid

echo "==> Running performance test..."
uv run python test_perf.py "${PY_ARGS[@]}"
