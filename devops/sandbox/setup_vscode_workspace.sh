#!/bin/bash

# ========== VS CODE INTEGRATION ==========
echo "Creating symlinks for VS Code IntelliSense..."

mkdir -p "/var/tmp/metta"

# Link site-packages for IntelliSense support
if [ -n "$CONDA_PREFIX" ]; then
  SITE_PACKAGES_PATH="$CONDA_PREFIX/lib/python3.11/site-packages"
  SYMLINK_PATH="/var/tmp/metta/conda-site-packages"
  ln -sf "$SITE_PACKAGES_PATH" "$SYMLINK_PATH"
  echo "Symlink created: $SYMLINK_PATH -> $SITE_PACKAGES_PATH"
else
  echo "WARNING: CONDA_PREFIX is not set. Make sure you've activated your conda environment."
  echo "Run 'conda activate metta' before running this script."
fi

# Link deps folder for external tools
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEPS_SOURCE="$PROJECT_ROOT/deps"
DEPS_SYMLINK="/var/tmp/metta/deps"

# Remove existing symlink if it exists
if [ -L "$DEPS_SYMLINK" ]; then
    rm -f "$DEPS_SYMLINK"
fi

if [ -d "$DEPS_SOURCE" ] && [ ! -L "$DEPS_SOURCE" ]; then
    ln -sf "$DEPS_SOURCE" "$DEPS_SYMLINK"
    echo "Symlink created: $DEPS_SYMLINK -> $DEPS_SOURCE"
else
    echo "WARNING: Could not find deps directory at $DEPS_SOURCE or it's already a symlink"
fi