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

echo "Upgrading pip..."
python -m pip install --upgrade pip

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# ========== Main Project ==========
cd "$SCRIPT_DIR/.."
echo "Installing main project requirements..."
pip install -r requirements.txt

# ========== CHECKOUT AND BUILD DEPENDENCIES ==========
echo "Calling devops/checkout_and_build script..."
bash "$SCRIPT_DIR/checkout_and_build.sh"

# ========== BUILD FAST_GAE ==========
echo "Building from setup.py (metta cython components)"
python setup.py build_ext --inplace

# ========== SANITY CHECK ==========
echo "Sanity check: verifying all local deps are importable"

# Add this right before the wandb_carbs check in your script
python -c "import sys; print('Python path:', sys.path); from carbs import CARBS; print('CARBS import worked')"

for dep in \
  "pufferlib" \
  "carbs" \
  "wandb_carbs"
do
  echo "Checking import for $dep..."
  python -c "import $dep; print('✅ Found {} at {}'.format('$dep', __import__('$dep').__file__))" || {
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
