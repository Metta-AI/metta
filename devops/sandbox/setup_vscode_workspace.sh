#!/bin/bash

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
