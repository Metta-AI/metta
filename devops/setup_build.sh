#!/bin/bash
# Setup script for Metta development environment
# This script installs all required dependencies and configures the environment
# Exit immediately if a command exits with a non-zero status
set -e

# Clear the deps built flag to ensure fresh checkout/build
rm -f deps/.built

if [ -f /.dockerenv ]; then
  export IS_DOCKER=true
else
  export IS_DOCKER=false
fi

# Check if we're in the correct conda environment
if [ "$CONDA_DEFAULT_ENV" != "metta" ] && [ -z "$CI" ] && [ -z "$IS_DOCKER" ]; then
  echo "WARNING: You must be in the 'metta' conda environment to run this script."
  echo "Please activate the correct environment with: \"conda activate metta\""
fi

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# ========== Main Project ==========
cd "$SCRIPT_DIR/.."

if [ -z "$CI" ] && [ -z "$IS_DOCKER" ]; then
  echo "Upgrading pip..."
  python -m pip install --upgrade pip
  
  echo -e "\n\nUninstalling all old python packages...\n\n"
  pip freeze | grep -v "^-e" > requirements_to_remove.txt
  pip uninstall -y -r requirements_to_remove.txt || echo "Some packages could not be uninstalled, continuing..."
  rm requirements_to_remove.txt
fi

# ========== CHECKOUT AND BUILD DEPENDENCIES ==========
echo -e "\n\nCalling devops/checkout_and_build script...\n\n"
bash "$SCRIPT_DIR/checkout_and_build.sh"

# ========== INSTALL PACKAGES BEFORE BUILD ==========
echo -e "\n\nInstalling main project requirements...\n\n"
pip install -c requirements.txt -r requirements.txt

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
"wandb_carbs"
do
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

if [ -z "$CI" ] && [ -z "$IS_DOCKER" ]; then
  # ========== VS CODE INTEGRATION ==========
  echo "Setting up VSCode integration..."
  source "$SCRIPT_DIR/sandbox/setup_vscode_workspace.sh"
  echo "✅ setup_build.sh completed successfully!"
fi