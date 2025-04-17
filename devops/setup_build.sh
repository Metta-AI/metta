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

# ========== VS CODE INTEGRATION ==========
# Create symlink for VS Code IntelliSense to find packages from Conda environment
# This allows VS Code to properly resolve imports from the conda environment
# Create symlink in /var/tmp with a standardized name
echo "Creating symlink for VS Code IntelliSense..."
if [ -n "$CONDA_PREFIX" ]; then
  SITE_PACKAGES_PATH="$CONDA_PREFIX/lib/python3.11/site-packages"
  SYMLINK_PATH="/var/tmp/metta/conda-site-packages"
  mkdir -p "/var/tmp/metta"
  ln -sf "$SITE_PACKAGES_PATH" "$SYMLINK_PATH"
  echo "Symlink created: $SYMLINK_PATH -> $SITE_PACKAGES_PATH"
else
  echo "WARNING: CONDA_PREFIX is not set. Make sure you've activated your conda environment."
  echo "Run 'conda activate metta' before running this script."
fi
