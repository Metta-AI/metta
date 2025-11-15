#!/usr/bin/env bash
set -euo pipefail

# This wrapper runs `uv sync` and then reinstalls PyTorch with the
# appropriate backend (CPU, cu12, cu13, etc.) using uv's torch backend
# auto-detection. It keeps heterogeneous machines in sync without
# hard-pinning CUDA-specific wheels in the lockfile.

BACKEND="${UV_TORCH_BACKEND:-auto}"
# Allow overriding the packages to reinstall; default to torch only.
TORCH_PACKAGES_STR="${UV_TORCH_PACKAGES:-torch==2.9.1}"

# Detect commands that shouldn't trigger PyTorch reinstalls (check/dry-run).
SKIP_REINSTALL=0
for arg in "$@"; do
  case "$arg" in
    --check|--check-exists|--dry-run|-h|--help)
      SKIP_REINSTALL=1
      ;;
  esac
done

echo "[uv-sync] Running uv sync $* (PyTorch backend: $BACKEND)"
uv sync "$@"

if [ "$SKIP_REINSTALL" -eq 1 ]; then
  exit 0
fi

# Split TORCH_PACKAGES_STR into an array (whitespace separated).
read -r -a TORCH_PACKAGES <<<"$TORCH_PACKAGES_STR"
if [ "${#TORCH_PACKAGES[@]}" -eq 0 ]; then
  exit 0
fi

PYTHON_BIN="${UV_PYTHON_PATH:-.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "[uv-sync] warning: expected venv Python at $PYTHON_BIN but it was not found" >&2
  echo "[uv-sync] skipping PyTorch reinstall" >&2
  exit 0
fi

if [ -n "${UV_TORCH_ALLOW_INVALID:-}" ]; then
  export UV_TORCH_ALLOW_INVALID
fi

echo "[uv-sync] Ensuring ${TORCH_PACKAGES[*]} is installed with backend $BACKEND"
UV_TORCH_BACKEND="$BACKEND" uv pip install --python "$PYTHON_BIN" "${TORCH_PACKAGES[@]}"
