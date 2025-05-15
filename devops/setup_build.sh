#!/bin/bash
# Setup script for Metta development environment
# This script installs all required dependencies and configures the environment
# Exit immediately if a command exits with a non-zero status
set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ -z "$CI" ]; then
  # ========== CLEAN BUILD ARTIFACTS ==========
  echo -e "\n\nCleaning build artifacts...\n\n"

  # Remove virtual environments
  if [ -d "$PROJECT_DIR/.venv" ]; then
    echo "Removing .venv virtual environment..."
    rm -rf "$PROJECT_DIR/.venv"
    echo "✅ Removed .venv virtual environment"
  fi
  if [ -d "$PROJECT_DIR/venv" ]; then
    echo "Removing venv virtual environment..."
    rm -rf "$PROJECT_DIR/venv"
    echo "✅ Removed venv virtual environment"
  fi

  # Clean root directory artifacts
  find "$PROJECT_DIR" -type f -name '*.so' -delete
  find "$PROJECT_DIR" -type d -name 'build' -exec rm -rf {} +
  echo "✅ Cleaned root directory build artifacts"

  # Clean mettagrid artifacts if directory exists
  if [ -d "$PROJECT_DIR/mettagrid" ]; then
    echo "Cleaning mettagrid build artifacts..."
    find "$PROJECT_DIR/mettagrid" -name "*.so" -type f -delete
    echo "✅ Removed .so files from mettagrid directory"
  fi
fi

if [ -f /.dockerenv ]; then
  export IS_DOCKER=true
else
  export IS_DOCKER=false
fi

# Verify uv is available
if ! command -v uv &> /dev/null; then
  # Try to find uv directly in common installation locations
  UV_BIN=""
  for possible_path in "$HOME/.cargo/bin/uv" "$HOME/.local/bin/uv"; do
    if [ -f "$possible_path" ]; then
      echo "Found uv at $possible_path but it's not in PATH"
      UV_BIN="$possible_path"
      break
    fi
  done

  if [ -n "$UV_BIN" ]; then
    echo "Will use uv at $UV_BIN directly"
    # Define a function to use the full path to uv
    uv() {
      "$UV_BIN" "$@"
    }
    export -f uv
  else
    echo "ERROR: uv command not found in PATH after installation attempts"
    echo "Current PATH: $PATH"
    echo "Please install uv manually: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "Then add uv to your PATH and try again"
    exit 1
  fi
fi

# ========== Main Project ==========
cd "$SCRIPT_DIR/.."

# Deactivate any active virtual environment first
if [ -n "$VIRTUAL_ENV" ]; then
  echo "Deactivating current virtual environment: $VIRTUAL_ENV"
  # This is a bit of a hack since 'deactivate' is a function in the activated environment
  # Using 'command' to temporarily disable the function behavior
  if [[ "$(type -t deactivate)" == "function" ]]; then
    deactivate
    echo "✅ Virtual environment deactivated"
  fi
fi

if [ -z "$CI" ] && [ -z "$IS_DOCKER" ]; then
  # We'll set up the virtual environment later
  echo -e "\n\nPreparing to set up environment...\n\n"
fi

# ========== INSTALL PACKAGES BEFORE BUILD ==========
echo -e "\n\nInstalling main project requirements...\n\n"

# Always create a fresh virtual environment
echo "Creating a virtual environment with uv..."

# Check and remove existing venv directories if needed
if [ -d ".venv" ]; then
  echo "Removing existing .venv directory..."
  rm -rf .venv
fi
if [ -d "venv" ]; then
  echo "Removing existing venv directory..."
  rm -rf venv
fi

echo "Creating new virtual environment..."
uv venv .venv --python 3.11.7 || {
  echo "Error: Failed to create virtual environment with uv command."
  exit 1
}

# Activate the virtual environment
if [[ -d ".venv" ]]; then
  # Activate the venv
  source .venv/bin/activate
  echo "✅ Virtual environment '.venv' created and activated"

  echo "Installing project requirements..."
  uv pip install -r requirements.txt || {
    if [ -n "$UV_BIN" ]; then
      echo "Retrying with direct path to uv: $UV_BIN"
      "$UV_BIN" pip install -r requirements.txt || {
        echo "❌ Failed to install packages. Please check the error message above."
        exit 1
      }
    else
      echo "❌ Failed to install packages. Please check the error message above."
      exit 1
    fi
  }
else
  echo "❌ Failed to create virtual environment with uv"
  exit 1
fi

echo -e "\n\nBuilding mettagrid...\n\n"
uv run --active --directory mettagrid python setup.py build_ext --inplace
uv pip install -e mettagrid

# ========== BUILD FAST_GAE ==========
echo -e "\n\nBuilding from setup.py (metta cython components)\n\n"
echo "Current working directory: $(pwd)"
uv run python setup.py build_ext --inplace

# ========== INSTALL SKYPILOT ==========
echo -e "\n\nInstalling Skypilot...\n\n"
uv tool install skypilot  --from 'skypilot[aws,vast,lambda]'

# ========== SANITY CHECK ==========
echo -e "\n\nSanity check: verifying all local deps are importable\n\n"

python -c "import sys; print('Python path:', sys.path);"

for dep in \
  "pufferlib" \
  "carbs" \
  "wandb_carbs"; do
  echo "Checking import for $dep..."
  python -c "import $dep; print('✅ Found {} at {}'.format('$dep', $dep.__file__))" || {
    echo "❌ Failed to import $dep"
    exit 1
  }
done

# Check for metta.rl.fast_gae.compute_gae
echo "Checking import for metta.rl.fast_gae.compute_gae..."
python -c "from metta.rl.fast_gae import compute_gae; print('✅ Found metta.rl.fast_gae.compute_gae')" || {
  echo "❌ Failed to import metta.rl.fast_gae.compute_gae"
  exit 1
}

# Check for mettagrid.mettagrid_env.MettaGridEnv
echo "Checking import for mettagrid.mettagrid_env.MettaGridEnv..."
python -c "from mettagrid.mettagrid_env import MettaGridEnv; print('✅ Found mettagrid.mettagrid_env.MettaGridEnv')" || {
  echo "❌ Failed to import mettagrid.mettagrid_env.MettaGridEnv"
  exit 1
}

# Check for mettagrid.mettagrid_c.MettaGrid
echo "Checking import for mettagrid.mettagrid_c.MettaGrid..."
python -c "from mettagrid.mettagrid_c import MettaGrid; print('✅ Found mettagrid.mettagrid_c.MettaGrid')" || {
  echo "❌ Failed to import mettagrid.mettagrid_c.MettaGrid"
  exit 1
}

if [ -z "$CI" ] && [ -z "$IS_DOCKER" ]; then
  # ========== VS CODE INTEGRATION ==========
  echo "Setting up VSCode integration..."
  source "$SCRIPT_DIR/sandbox/setup_vscode_workspace.sh"
  echo "✅ setup_build.sh completed successfully!"
fi

# ========== INSTALL METTASCOPE ==========
bash "mettascope/install.sh"
