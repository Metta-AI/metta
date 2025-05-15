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

if [ -z "$CI" ]; then
  # Verify uv is available
  if ! command -v uv &> /dev/null; then
    echo "ERROR: uv command not found in PATH after installation attempts"
    echo "Current PATH: $PATH"
    echo "Please install uv manually: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "Then add uv to your PATH and try again"
    exit 1
  fi
fi

# Check if we're in a virtual environment
if [ -z "${VIRTUAL_ENV}" ]; then
  echo "Warning: Not running in a virtual environment. Checking for .venv directory..."

  # Check the project root directory for .venv
  if [ -d "../.venv" ]; then
    echo "Found .venv directory, activating it..."
    if [ -f "../.venv/bin/activate" ]; then
      source "../.venv/bin/activate"
      echo "Activated virtual environment: $VIRTUAL_ENV"
    else
      echo "Warning: Found .venv directory but couldn't locate activation script."
      exit 1
    fi
  else
    echo "Warning: No virtual environment found. This may cause build issues."
  fi
else
  echo "Using virtual environment: $VIRTUAL_ENV"
fi

echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo "Current directory: $(pwd)"
echo "Python path: $(python -c 'import sys; print(sys.path)')"

# Go to the project root directory
cd "$SCRIPT_DIR/.."

# Install mettagrid requirements
uv pip install -r requirements.txt

# Navigate to mettagrid directory
cd mettagrid


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
uv pip install -e .

echo "========== mettagrid rebuild complete =========="
