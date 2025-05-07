#!/bin/bash
# Setup script for Metta development environment
# This script installs all required dependencies and configures the environment
# Exit immediately if a command exits with a non-zero status
set -e

# Clear the deps built flag to ensure fresh checkout/build
rm -f deps/.built

# Check if we're in the correct conda environment
if [ "$CONDA_DEFAULT_ENV" != "metta" ] && [ -z "$CI" ]; then
  echo "WARNING: You must be in the 'metta' conda environment to run this script."
  echo "Please activate the correct environment with: \"conda activate metta\""
fi

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# ========== Main Project ==========
cd "$SCRIPT_DIR/.."

if [ -z "$CI" ]; then
  echo -e "\n\nUpgrading pip...\n\n"
  python -m pip install --upgrade pip

  echo -e "\n\nUninstalling all old python packages...\n\n"
  pip freeze | grep -v "^-e" > requirements_to_remove.txt
  pip uninstall -y -r requirements_to_remove.txt
  rm requirements_to_remove.txt
fi

# ========== INSTALL PACKAGES BEFORE BUILD ==========
echo -e "\n\nInstalling main project requirements...\n\n"
pip install pettingzoo==1.25.0 --no-deps  # Install pettingzoo without dependencies
pip install -c requirements.txt -r requirements.txt

# ========== CHECKOUT AND BUILD DEPENDENCIES ==========
echo -e "\n\nCalling devops/checkout_and_build script...\n\n"
bash "$SCRIPT_DIR/checkout_and_build.sh"

# ========== BUILD FAST_GAE ==========
echo -e "\n\nBuilding from setup.py (metta cython components)\n\n"
python setup.py build_ext --inplace

# ========== SANITY CHECK ==========
echo -e "\n\nSanity check: verifying all local deps are importable\n\n"
# Add this right before the wandb_carbs check in your script
python -c "import sys; print('Python path:', sys.path); from carbs import CARBS; print('CARBS import worked')"

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

if [ -z "$CI" ]; then
  # ========== VS CODE INTEGRATION ==========
  echo "Setting up VSCode integration..."
  source "$SCRIPT_DIR/sandbox/setup_vscode_workspace.sh"
  echo "✅ setup_build.sh completed successfully!"
fi