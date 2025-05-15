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

# ========== Check for uv ==========
if ! command -v uv &> /dev/null && [ -z "$CI" ] && [ -z "$IS_DOCKER" ]; then
  echo "uv is not installed. Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Make sure uv is in the PATH
  export PATH="$HOME/.cargo/bin:$PATH"
fi

# ========== Main Project ==========
cd "$SCRIPT_DIR/.."

if [ -z "$CI" ] && [ -z "$IS_DOCKER" ]; then
  echo -e "\n\nCreating virtual environment with uv...\n\n"
  uv venv --python 3.11.7
  
  # Detect if we're in the virtual environment - uv will try to activate it, but in scripts we need to do it manually
  if [[ -d ".venv" ]]; then
    # Activate the venv
    if [[ -f ".venv/bin/activate" ]]; then
      source .venv/bin/activate
    elif [[ -f ".venv/Scripts/activate" ]]; then
      source .venv/Scripts/activate
    fi
    echo "✅ Virtual environment created and activated"
  else
    echo "❌ Failed to create virtual environment with uv"
    exit 1
  fi

  echo -e "\n\nCleaning any existing packages...\n\n"
  uv pip uninstall --all
fi

# ========== INSTALL PACKAGES BEFORE BUILD ==========
echo -e "\n\nInstalling main project requirements...\n\n"
uv pip install -r requirements.txt

echo -e "\n\nCalling devops/build_mettagrid script...\n\n"
bash "$SCRIPT_DIR/build_mettagrid.sh"

# ========== BUILD FAST_GAE ==========
echo -e "\n\nBuilding from setup.py (metta cython components)\n\n"
echo "Current working directory: $(pwd)"
python setup.py build_ext --inplace

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
