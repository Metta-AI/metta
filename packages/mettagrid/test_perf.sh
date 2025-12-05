#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

echo "==> Rebuilding mettagrid package (builds .so via bazel_build.py)..."
cd ../..
uv sync --reinstall-package mettagrid
cd packages/mettagrid

echo "==> Running performance test..."
uv run python test_perf.py "$@"
