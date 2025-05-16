#!/bin/bash
# Setup script for Metta development environment
# This script installs all required dependencies and configures the environment
# Exit immediately if a command exits with a non-zero status
set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ -z "$CI" ]; then
  # ========== CLEAN BUILD ARTIFACTS ==========
  echo -e "\nCleaning build artifacts..."

  # Clean up old venvs
  is_uv_venv() {
      local venv_path="$1"
      # Look for uv marker files in the venv
      if [ -f "$venv_path/pyvenv.cfg" ] && grep -q "uv" "$venv_path/pyvenv.cfg"; then
          return 0  # It's a UV venv
      elif [ -d "$venv_path/.uv" ]; then
          return 0  # It has a .uv directory
      else
          return 1  # Not a UV venv
      fi
  }
  
  # Remove standard virtual environments but preserve UV venvs
  VENV_PATHS=(".venv" "venv" ".env" "env" "virtualenv" ".virtualenv")

  # Process each potential virtual environment
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
  
  # Report any residual conda environments
  if command -v conda &> /dev/null; then
      echo "Checking for conda environments associated with this project..."
      
      # Extract project name from directory
      PROJECT_NAME=$(basename "$PROJECT_DIR")
      
      # List all conda environments and filter for ones that might be related to this project
      CONDA_ENVS=$(conda env list | grep "$PROJECT_NAME" | awk '{print $1}')
      
    if [ -n "$CONDA_ENVS" ]; then
        echo "⚠️  Found the following conda environments that might be related to this project:"
        echo "$CONDA_ENVS"
        echo "⚠️  You may want to manually remove these if they're no longer needed (conda env remove -n ENV_NAME)"
    fi
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

# Only deactivate an existing virtual environment if it's not our target environment
if [ -n "$VIRTUAL_ENV" ]; then
  if [[ "$VIRTUAL_ENV" != *"$PROJECT_DIR/.venv"* ]]; then
    echo -e "\nDeactivating current virtual environment: $VIRTUAL_ENV..."
    # This is a bit of a hack since 'deactivate' is a function in the activated environment
    if [[ "$(type -t deactivate)" == "function" ]]; then
      deactivate
      echo "✅ Virtual environment deactivated"
    fi
  else
    echo "Already using the project's virtual environment"
  fi
fi

# ========== INSTALL PACKAGES BEFORE BUILD ==========
echo -e "\nInstalling main project requirements...\n"

# Check if a valid UV virtual environment already exists
if [ -d ".venv" ] && is_uv_venv ".venv"; then
  echo "Found existing UV virtual environment in .venv"
  
  # Activate the existing venv
  source .venv/bin/activate
  echo "✅ Existing virtual environment '.venv' activated"
  
  # Check Python version in the existing environment to ensure it's what we need
  CURRENT_PYTHON_VERSION=$(python -c "import platform; print(platform.python_version())")
  REQUIRED_PYTHON_VERSION="3.11.7"
  
  if [[ "$CURRENT_PYTHON_VERSION" == "$REQUIRED_PYTHON_VERSION"* ]]; then
    echo "✅ Python version $CURRENT_PYTHON_VERSION meets requirements"
  else
    echo "⚠️ Current Python version ($CURRENT_PYTHON_VERSION) does not match required version ($REQUIRED_PYTHON_VERSION)"
    echo "Recreating virtual environment with the correct Python version..."
    
    # Deactivate current environment
    deactivate
    
    # Delete the old environment
    echo -e "\nRemoving old virtual environment..."
    rm -rf .venv
    echo "✅ Old virtual environment removed"
    
    # Create a new environment with the correct Python version
    uv venv .venv --python $REQUIRED_PYTHON_VERSION || {
      echo "Error: Failed to create new virtual environment with uv command."
      exit 1
    }
    
    # Activate the new environment
    source .venv/bin/activate
    echo "✅ New virtual environment '.venv' created and activated with Python $REQUIRED_PYTHON_VERSION"
  fi
else
  # No valid UV environment exists, create a new one
  echo -e "\nCreating new virtual environment with uv..."
  uv venv .venv --python 3.11.7 || {
    echo "Error: Failed to create virtual environment with uv command."
    exit 1
  }
  
  # Activate the venv
  source .venv/bin/activate
  echo "✅ Virtual environment '.venv' created and activated"
fi

# Update/install packages regardless of whether we're using an existing or new environment
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
make build
cd ..
uv pip install -e mettagrid

# ========== BUILD FAST_GAE ==========
echo -e "\nBuilding from setup.py (metta cython components)..."
make build

# ========== INSTALL SKYPILOT ==========
echo -e "\nInstalling Skypilot..."
uv tool install skypilot  --from 'skypilot[aws,vast,lambda]'

# ========== SANITY CHECK ==========
python -c "import sys; print('Python path:', sys.path);"

for dep in \
  "pufferlib" \
  "carbs" \
  "wandb_carbs"; do
  echo -e "\nChecking import for $dep..."
  python -c "import $dep; print('✅ Found {} at {}'.format('$dep', $dep.__file__))" || {
    echo "❌ Failed to import $dep"
    exit 1
  }
done

# Check for metta.rl.fast_gae.compute_gae
echo -e "\nChecking import for metta.rl.fast_gae.compute_gae..."
python -c "from metta.rl.fast_gae import compute_gae; print('✅ Found metta.rl.fast_gae.compute_gae')" || {
  echo "❌ Failed to import metta.rl.fast_gae.compute_gae"
  exit 1
}

# Check for mettagrid.mettagrid_env.MettaGridEnv
echo -e "\nChecking import for mettagrid.mettagrid_env.MettaGridEnv..."
python -c "from mettagrid.mettagrid_env import MettaGridEnv; print('✅ Found mettagrid.mettagrid_env.MettaGridEnv')" || {
  echo "❌ Failed to import mettagrid.mettagrid_env.MettaGridEnv"
  exit 1
}

# Check for mettagrid.mettagrid_c.MettaGrid
echo -e "\nChecking import for mettagrid.mettagrid_c.MettaGrid..."
python -c "from mettagrid.mettagrid_c import MettaGrid; print('✅ Found mettagrid.mettagrid_c.MettaGrid')" || {
  echo "❌ Failed to import mettagrid.mettagrid_c.MettaGrid"
  exit 1
}

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
