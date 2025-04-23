#!/bin/bash
# Setup script for Metta development environment
# This script installs all required dependencies and configures the environment

# Exit immediately if a command exits with a non-zero status
set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Create and enter deps directory for all external dependencies
echo "Creating deps directory..."
mkdir -p deps
cd deps

# ========== FAST_GAE ==========
if [ ! -d "fast_gae" ]; then
  echo "Cloning fast_gae into $(pwd)"
  git clone https://github.com/Metta-AI/fast_gae.git
fi
cd fast_gae
echo "Updating fast_gae..."
git pull
echo "Building fast_gae into $(pwd)"
python setup.py build_ext --inplace
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
git pull
cd ..

# ========== METTAGRID ==========
cd mettagrid
echo "Building mettagrid into $(pwd)"
python setup.py build_ext --inplace
cd ..

# ========== CARBS ==========
if [ ! -d "carbs" ]; then
  echo "Cloning carbs into $(pwd)"
  #git clone https://github.com/imbue-ai/carbs.git
  git clone https://github.com/kywch/carbs.git
fi
cd carbs
echo "Updating carbs..."
git pull
cd ..

# ========== WANDB_CARBS ==========
if [ ! -d "wandb_carbs" ]; then
  echo "Cloning wandb_carbs into $(pwd)"
  git clone https://github.com/Metta-AI/wandb_carbs.git
fi
cd wandb_carbs
echo "Updating wandb_carbs..."
git pull
cd ..

# ========== Metta requirements.txt ==========
cd "$SCRIPT_DIR/.."
echo "Installing metta python requirements..."
pip install -r requirements.txt

echo "Sanity check: trying to import pufferlib"
python -c "import pufferlib; print('✅ Found pufferlib at', pufferlib.__file__)"

# TODO -- ideally we can find a way to skip this step when we are not on a user's machine
# for now we are including this here as a convenience because the README and other places
# tell people to setup their workspace using ./devops/setup_build.sh

# ========== VS CODE INTEGRATION ==========
echo "Setting up VSCode integration..."
source "$SCRIPT_DIR/sandbox/setup_vscode_workspace.sh"
