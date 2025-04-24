#!/bin/bash
# Setup script for Metta development environment
# This script installs all required dependencies and configures the environment

# Exit immediately if a command exits with a non-zero status
set -e

# Check if we're in the correct conda environment
if [ "$CONDA_DEFAULT_ENV" != "metta" ]; then
    echo "Error: You must be in the 'metta' conda environment to run this script."
    echo "Please activate the correct environment with: \"conda activate metta\""
    exit 1
fi

echo "Upgrading pip..."
python -m pip install --upgrade pip

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# ========== Main Project ==========
cd "$SCRIPT_DIR/.."
echo "Installing main project requirements..."
pip install -r requirements.txt

# Create and enter deps directory for all external dependencies
echo "Creating deps directory..."
mkdir -p deps
cd deps

# ========== METTAGRID ==========
cd mettagrid
echo "Building mettagrid into $(pwd)"
python setup.py build_ext --inplace
pip install -e .
export PYTHONPATH="$PYTHONPATH:$(pwd)"
echo "Updated PYTHONPATH: $PYTHONPATH"

cd ..

# ========== FAST_GAE ==========
if [ ! -d "fast_gae" ]; then
  echo "Cloning fast_gae into $(pwd)"
  git clone https://github.com/Metta-AI/fast_gae.git
fi
cd fast_gae
echo "Fetching fast_gae into $(pwd)"
git fetch
echo "Checking out main into $(pwd)"
git checkout main
echo "Updating fast_gae..."
git pull || echo "⚠️ Warning: git pull failed, possibly shallow clone or detached HEAD"
echo "Building fast_gae into $(pwd)"
python setup.py build_ext --inplace
pip install -e .
export PYTHONPATH="$PYTHONPATH:$(pwd)"
echo "Updated PYTHONPATH: $PYTHONPATH"

cd ..


# ========== PUFFERLIB ==========
if [ ! -d "pufferlib" ]; then
  echo "Cloning pufferlib into $(pwd)"
  git clone https://github.com/Metta-AI/pufferlib.git
fi
cd pufferlib
echo "Fetching pufferlib into $(pwd)"
git fetch
echo "Checking out metta into $(pwd)"
git checkout metta
echo "Updating pufferlib..."
git pull || echo "⚠️ Warning: git pull failed, possibly shallow clone or detached HEAD"
echo "Installing pufferlib (in normal mode)"
pip install .
export PYTHONPATH="$PYTHONPATH:$(pwd)"
echo "Updated PYTHONPATH: $PYTHONPATH"

cd ..

# ========== CARBS ==========
if [ ! -d "carbs" ]; then
  echo "Cloning carbs into $(pwd)"
  #git clone https://github.com/imbue-ai/carbs.git
  git clone https://github.com/kywch/carbs.git
fi
cd carbs
echo "Fetching carbs into $(pwd)"
git fetch
echo "Checking out main branch for carbs"
git checkout main
echo "Updating carbs..."
git pull || echo "⚠️ Warning: git pull failed, possibly shallow clone or detached HEAD"
pip install -e .
export PYTHONPATH="$PYTHONPATH:$(pwd)"
echo "Updated PYTHONPATH: $PYTHONPATH"

cd ..

# ========== WANDB_CARBS ==========
if [ ! -d "wandb_carbs" ]; then
  echo "Cloning wandb_carbs into $(pwd)"
  git clone https://github.com/Metta-AI/wandb_carbs.git
fi
cd wandb_carbs
echo "Fetching wandb_carbs into $(pwd)"
git fetch
echo "Checking out main branch for wandb_carbs"
git checkout main
echo "Updating wandb_carbs..."
git pull || echo "⚠️ Warning: git pull failed, possibly shallow clone or detached HEAD"
pip install -e .
export PYTHONPATH="$PYTHONPATH:$(pwd)"
echo "Updated PYTHONPATH: $PYTHONPATH"

cd ..

# ========== SANITY CHECK ==========
echo "Sanity check: verifying all local deps are importable"

# Add this right before the wandb_carbs check in your script
python -c "import sys; print('Python path:', sys.path); from carbs import CARBS; print('CARBS import worked')"

for dep in \
  "pufferlib" \
  "fast_gae" \
  "mettagrid" \
  "carbs" \
  "wandb_carbs"
do
  echo "Checking import for $dep..."
  python -c "import $dep; print('✅ Found {} at {}'.format('$dep', __import__('$dep').__file__))" || {
    echo "❌ Failed to import $dep"
    exit 1
  }
done


# TODO -- ideally we can find a way to skip this step when we are not on a user's machine
# for now we are including this here as a convenience because the README and other places
# tell people to setup their workspace using ./devops/setup_build.sh

# ========== VS CODE INTEGRATION ==========
echo "Setting up VSCode integration..."
source "$SCRIPT_DIR/sandbox/setup_vscode_workspace.sh"

echo "✅ setup_build.sh completed successfully!"