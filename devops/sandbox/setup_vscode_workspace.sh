#!/bin/bash

# ========== VS CODE INTEGRATION ==========
echo "Creating symlinks for VS Code IntelliSense..."

mkdir -p "/var/tmp/metta"

# Get the current directory (project root)
PROJECT_DIR="$(pwd)"

# Check for uv virtual environment
if [ -d "$PROJECT_DIR/.venv" ]; then
  # Determine the site-packages path based on OS
  if [ -d "$PROJECT_DIR/.venv/lib/python3.11/site-packages" ]; then
    # macOS/Linux path
    SITE_PACKAGES_PATH="$PROJECT_DIR/.venv/lib/python3.11/site-packages"
  elif [ -d "$PROJECT_DIR/.venv/Lib/site-packages" ]; then
    # Windows path
    SITE_PACKAGES_PATH="$PROJECT_DIR/.venv/Lib/site-packages"
  fi

  if [ -n "$SITE_PACKAGES_PATH" ]; then
    SYMLINK_PATH="/var/tmp/metta/venv-site-packages"
    ln -sf "$SITE_PACKAGES_PATH" "$SYMLINK_PATH"
    echo "Symlink created: $SYMLINK_PATH -> $SITE_PACKAGES_PATH"
  else
    echo "WARNING: Could not determine site-packages path in .venv directory"
  fi
else
  echo "WARNING: No .venv directory found. Make sure you've set up a virtual environment with uv."
  echo "Run './devops/setup_build.sh' to create the virtual environment."
fi
