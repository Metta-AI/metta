#!/bin/bash
# Setup script for Metta development environment
# This script installs all required dependencies and configures the environment

# Exit immediately if a command exits with a non-zero status
set -e
set -o pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")" # (root)/devops
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"        # (root)
cd "$PROJECT_DIR"

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
  export IS_DOCKER=1
else
  export IS_DOCKER=''
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

VENV_PATH=".venv"

# ========== CLEAN BUILD ==========
if [ "$CLEAN" -eq 1 ]; then

  make clean

  # deactivate the venv
  echo -e "\nDeactivating current virtual environment: $VIRTUAL_ENV"
  # This is a bit of a hack since 'deactivate' is a function in the activated environment
  # Using 'command' to temporarily disable the function behavior
  if [[ "$(type -t deactivate)" == "function" ]]; then
    deactivate
    echo "✅ Virtual environment deactivated"
  fi

  # Remove the virtual environment
  if [ -d .venv ]; then
    echo "Removing .venv virtual environment..."
    rm -rf .venv
    echo "✅ Removed .venv virtual environment"
  fi
fi

# ========== CREATE VENV ==========
# Check if we're already in a UV venv
if ! is_uv_venv; then
  # Check if a virtual environment exists but is not activated
  if [ -d "$VENV_PATH" ]; then
    echo "⚠️ Existing environment is not a UV environment, recreating it"

    # Remove the existing environment
    rm -rf "$VENV_PATH"
  fi

  # Create a new environment
  echo "Creating new virtual environment..."
  uv venv "$VENV_PATH" || {
    echo "Error: Failed to create virtual environment with uv command."
    exit 1
  }
  echo "✅ Virtual environment '$VENV_PATH' created and activated"

  # Activate the environment
  source "$VENV_PATH/bin/activate"
fi

if [ -z "$CI" ]; then
  # ========== REPORT RESIDUAL CONDA VENV ==========
  if command -v conda &> /dev/null; then
    echo "Checking for conda environments associated with this project..."
    PROJECT_NAME=$(basename "$(pwd)")
    CONDA_ENVS=$(conda env list | grep "$PROJECT_NAME" | awk '{print $1}')
    if [ -n "$CONDA_ENVS" ]; then
      echo "⚠️  Found the following conda environments that might be related to this project:"
      echo "$CONDA_ENVS"
      echo "⚠️  You may want to manually remove these if they're no longer needed (conda env remove -n ENV_NAME)"
    fi
  fi
fi

# ========== BUILD AND INSTALL ==========

echo -e "\nInstalling project requirements..."
uv pip install -r requirements.txt

echo -e "\nInstalling Metta..."
uv pip install -e .

echo -e "\nInstalling MettaGrid..."
uv pip --directory mettagrid install -e .

PYTHON="uv run --active python"

# ========== SANITY CHECK ==========
echo -e "\nSanity check: verifying all local deps are importable..."

for dep in \
  "pufferlib" \
  "carbs" \
  "metta.rl.fast_gae.fast_gae" \
  "mettagrid.mettagrid_env" \
  "mettagrid.mettagrid_c" \
  "wandb_carbs"; do
  uv run python -c "import $dep; print('✅ Found {} at {}'.format('$dep', $dep.__file__))" || {
    echo "❌ Failed to import $dep"
    exit 1
  }
done

if [ -z "$CI" ] && [ -z "$IS_DOCKER" ]; then
  # ========== VS CODE INTEGRATION ==========
  echo -e "\nSetting up VSCode integration..."
  source "./devops/sandbox/setup_vscode_workspace.sh"

  # ========== INSTALL METTASCOPE ==========
  echo -e "\nSetting up MettaScope..."
  bash "mettascope/install.sh"

  # ========== CHECK AWS TOKEN SETUP ==========
  echo -e "\nSetting up AWS access..."
  bash "devops/aws/setup_aws_profiles.sh"

  echo -e "\nInstalling Skypilot..."
  bash "devops/skypilot/install.sh"

  echo "✅ setup_build.sh completed successfully!"
  echo -e "Activate virtual environment with: \033[32;1msource $VENV_PATH/bin/activate\033[0m"
fi
