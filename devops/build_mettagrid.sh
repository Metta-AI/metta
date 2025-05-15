#!/bin/bash

# This script rebuilds mettagrid without rebuilding other dependencies

# Exit immediately if a command exits with a non-zero status
set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Parse command line arguments
CLEAN=0
for arg in "$@"; do
  case $arg in
    --clean)
      CLEAN=1
      shift
      ;;
  esac
done

# Display appropriate header based on clean flag
if [ "$CLEAN" -eq 1 ]; then
  echo "========== Rebuilding mettagrid (clean) =========="
else
  echo "========== Rebuilding mettagrid =========="
fi

echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo "Current directory: $(pwd)"
echo "Python path: $(python -c 'import sys; print(sys.path)')"

# Go to the project root directory
cd "$SCRIPT_DIR/.."

# Navigate to mettagrid directory
cd mettagrid

if [ -z "$CI" ]; then
  # ========== Check for uv ==========
  if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Make sure uv is in the PATH
    export PATH="$HOME/.cargo/bin:$PATH"
  fi

  echo "Installing mettagrid requirements..."
  uv pip install -r requirements.txt
fi

echo "Building mettagrid in $(pwd)"

# Clean build artifacts only if --clean flag is specified
if [ "$CLEAN" -eq 1 ]; then
  echo "Cleaning previous build artifacts..."
  rm -rf build
  find . -name "*.so" -delete
  echo "Clean completed."
else
  echo "Skipping clean (use --clean to remove previous build artifacts)"
fi

# Rebuild mettagrid
echo "Rebuilding mettagrid..."
python setup.py build_ext --inplace

# Reinstall in development mode
echo "Reinstalling mettagrid in development mode..."
if command -v uv &> /dev/null; then
  uv pip install -e .
else
  pip install -e .
fi

echo "========== mettagrid rebuild complete =========="
