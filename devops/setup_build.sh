#!/bin/bash
# Setup script for Metta development environment
# This script installs all required dependencies and configures the environment

# Exit immediately if a command exits with a non-zero status
set -e

# Install base requirements
echo "Installing metta python requirements..."
pip install -r requirements.txt

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
echo "Installing fast_gae into $(pwd)"
pip install -e .
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
echo "Installing pufferlib into $(pwd)"
pip install -e .
echo "Stashing pufferlib into $(pwd)"
git stash
cd ..

# ========== METTAGRID ==========
if [ ! -d "mettagrid" ]; then
  echo "Cloning mettagrid into $(pwd)"
  git clone https://github.com/Metta-AI/mettagrid.git
fi
cd mettagrid
echo "Fetching mettagrid into $(pwd)"
git fetch
# Check out the specified reference
if [ -n "$METTAGRID_REF" ]; then
  echo "Checking out mettagrid reference: $METTAGRID_REF"
  git checkout "$METTAGRID_REF"
fi
echo "Installing mettagrid python requirements..."
pip install -r requirements.txt
echo "Building mettagrid into $(pwd)"
python setup.py build_ext --inplace
echo "Installing mettagrid into $(pwd)"
pip install -e .
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
echo "Installing carbs into $(pwd)"
pip install -e .
cd ..

# ========== WANDB_CARBS ==========
if [ ! -d "wandb_carbs" ]; then
  echo "Cloning wandb_carbs into $(pwd)"
  git clone https://github.com/Metta-AI/wandb_carbs.git
fi
cd wandb_carbs
echo "Updating wandb_carbs..."
git pull
echo "Installing wandb_carbs into $(pwd)"
pip install -e .
cd ..

# TODO -- ideally we can find a way to skip this step when we are not on a user's machine
# for now we are including this here as a convenience because the README and other places
# tell people to setup their workspace using ./devops/setup_build.sh

# ========== VS CODE INTEGRATION ==========
echo "Setting up VSCode integration..."
source "$(dirname "$0")/sandbox/setup_vscode_workspace.sh"