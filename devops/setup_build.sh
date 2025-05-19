#!/bin/bash
# Setup script for Metta development environment
# This script installs all required dependencies and configures the environment

# Exit immediately if a command exits with a non-zero status
set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

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

# check if we're in docker
if [ -f /.dockerenv ]; then
  export IS_DOCKER=true
else
  export IS_DOCKER=false
fi

# Define a function to check if we're in a UV virtual environment
is_uv_venv() {
  # Check if we're in a virtual environment
  if [ -z "$VIRTUAL_ENV" ]; then
    return 1 # Not in any virtual environment
  fi

  # Check if it's a UV virtual environment by looking for UV marker files
  if [ -f "$VIRTUAL_ENV/pyvenv.cfg" ] && grep -q "uv" "$VIRTUAL_ENV/pyvenv.cfg"; then
    return 0 # It's a UV venv
  elif [ -d "$VIRTUAL_ENV/.uv" ]; then
    return 0 # It has a .uv directory
  else
    return 1 # Not a UV venv
  fi
}

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

# Required Python version
REQUIRED_PYTHON_VERSION="3.11.7"

# ========== CLEAN BUILD ==========
if [ "$CLEAN" -eq 1 ]; then
  # first deactivate the venv
  echo -e "\nDeactivating current virtual environment: $VIRTUAL_ENV"
  # This is a bit of a hack since 'deactivate' is a function in the activated environment
  # Using 'command' to temporarily disable the function behavior
  if [[ "$(type -t deactivate)" == "function" ]]; then
    deactivate
    echo "✅ Virtual environment deactivated"
  fi

  echo -e "\nCleaning build artifacts..."

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

  echo -e "\nCreating a new virtual environment with uv..."

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
  uv venv .venv --python $REQUIRED_PYTHON_VERSION || {
    echo "Error: Failed to create virtual environment with uv command."
    exit 1
  }

  # Activate the virtual environment
  if [[ -d ".venv" ]]; then
    # Activate the venv
    source .venv/bin/activate
    echo "✅ Virtual environment '.venv' created and activated"
  else
    echo "❌ Failed to create virtual environment with uv"
    exit 1
  fi
fi

# ========== REPORT RESIDUAL CONDA VENV ==========
if command -v conda &> /dev/null; then
  echo "Checking for conda environments associated with this project..."
  PROJECT_NAME=$(basename "$PROJECT_DIR")
  CONDA_ENVS=$(conda env list | grep "$PROJECT_NAME" | awk '{print $1}')
  if [ -n "$CONDA_ENVS" ]; then
    echo "⚠️  Found the following conda environments that might be related to this project:"
    echo "$CONDA_ENVS"
    echo "⚠️  You may want to manually remove these if they're no longer needed (conda env remove -n ENV_NAME)"
  fi
fi

# ========== REPORT Non-UV VENV ==========
VENV_PATHS=(".venv" "venv" ".env" "env" "virtualenv" ".virtualenv")
for venv_name in "${VENV_PATHS[@]}"; do
  venv_path="$PROJECT_DIR/$venv_name"
  if [ -d "$venv_path" ]; then
    if is_uv_venv "$venv_path"; then
      echo "Preserving $venv_name as it appears to be a UV virtual environment"
    else
      echo "Removing $venv_name virtual environment..."
      rm -rf "$venv_path"
      echo "✅ Removed $venv_name virtual environment"
    fi
  fi
done

# ========== CLEAN ALL BUILD ARTIFACTS ==========
if [ "$CLEAN" -eq 1 ]; then
  make clean
fi

# ========== Main Project ==========
cd "$SCRIPT_DIR/.."

# Install packages
echo -e "\nInstalling project requirements..."
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

# ========== BUILD METTAGRID ==========
echo -e "\nBuilding mettagrid..."
cd mettagrid
if [ "$CLEAN" -eq 1 ]; then
  make clean
  make install-dependencies
fi
make build
cd ..
uv pip install mettagrid

# ========== BUILD FAST_GAE ==========
echo -e "\nBuilding FastGAE..."
make build

# ========== INSTALL SKYPILOT ==========
echo -e "\nInstalling Skypilot..."
uv tool install skypilot --from 'skypilot[aws,vast,lambda]'

if [ "$CLEAN" -eq 1 ]; then
  PYTHON="uv run -- python"

  # ========== SANITY CHECK ==========
  echo -e "\nSanity check: verifying all local deps are importable..."

  $PYTHON -c "import sys; print('Python path:', sys.path);"

  for dep in \
    "pufferlib" \
    "carbs" \
    "wandb_carbs"; do
    echo -e "\nChecking import for $dep..."
    $PYTHON -c "import $dep; print('✅ Found {} at {}'.format('$dep', $dep.__file__))" || {
      echo "❌ Failed to import $dep"
      exit 1
    }
  done

  # Check for metta.rl.fast_gae.compute_gae
  echo -e "\nChecking import for metta.rl.fast_gae.compute_gae..."
  $PYTHON -c "from metta.rl.fast_gae import compute_gae; print('✅ Found metta.rl.fast_gae.compute_gae')" || {
    echo "❌ Failed to import metta.rl.fast_gae.compute_gae"
    exit 1
  }

  # Check for mettagrid.mettagrid_env.MettaGridEnv
  echo -e "\nChecking import for mettagrid.mettagrid_env.MettaGridEnv..."
  $PYTHON -c "from mettagrid.mettagrid_env import MettaGridEnv; print('✅ Found mettagrid.mettagrid_env.MettaGridEnv')" || {
    echo "❌ Failed to import mettagrid.mettagrid_env.MettaGridEnv"
    exit 1
  }

  # Check for mettagrid.mettagrid_c.MettaGrid
  echo -e "\nChecking import for mettagrid.mettagrid_c.MettaGrid..."
  $PYTHON -c "from mettagrid.mettagrid_c import MettaGrid; print('✅ Found mettagrid.mettagrid_c.MettaGrid')" || {
    echo "❌ Failed to import mettagrid.mettagrid_c.MettaGrid"
    exit 1
  }
fi

if [ -z "$CI" ] && [ -z "$IS_DOCKER" ]; then
  # ========== VS CODE INTEGRATION ==========
  echo -e "\nSetting up VSCode integration..."
  source "$SCRIPT_DIR/sandbox/setup_vscode_workspace.sh"
  echo "✅ setup_build.sh completed successfully!"

  # ========== INSTALL METTASCOPE ==========
  echo -e "\nSetting up MettaScope..."
  bash "mettascope/install.sh"

  # ========== CHECK AWS TOKEN SETUP ==========
  echo -e "\nSetting up AWS access..."
  bash "devops/aws/setup_aws_profiles.sh"
fi
