#!/bin/bash
# Setup script for Metta development environment
# This script installs all required dependencies and configures the environment

# Exit immediately if a command exits with a non-zero status
set -e
set -o pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")" # (root)/devops
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"        # (root)
cd "$PROJECT_DIR"

# Parse command line arguments
CLEAN=0
for arg in "$@"; do
  case $arg in
    --clean)
      CLEAN=1
      shift
      ;;
  esac
done

if [ -f /.dockerenv ]; then
  export IS_DOCKER=1
fi

# Verify uv is available
if ! command -v uv &> /dev/null; then
  # Try to find uv directly in common installation locations
  UV_BIN=""
  for possible_path in "$HOME/.cargo/bin/uv" "$HOME/.local/bin/uv"; do
    if [ -f "$possible_path" ]; then
      echo "Found uv at $possible_path but it's not in PATH"
      UV_BIN="$possible_path"
      break
    fi
  done

  if [ -n "$UV_BIN" ]; then
    echo "Will use uv at $UV_BIN directly"
    # Define a function to use the full path to uv
    uv() {
      "$UV_BIN" "$@"
    }
    export -f uv
  else
    echo "ERROR: uv command not found in PATH after installation attempts"
    echo "Current PATH: $PATH"
    echo "Please install uv manually: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "Then add uv to your PATH and try again"
    exit 1
  fi
fi

VENV_PATH=".venv"

# ========== CLEAN BUILD ==========
if [ "$CLEAN" -eq 1 ]; then

  make clean

  if [ -d .venv ]; then
    echo "Removing .venv virtual environment..."
    rm -rf .venv
    echo "✅ Removed .venv virtual environment"
  fi
fi

# ========== BUILD AND INSTALL ==========

echo -e "\nInstalling Metta..."
uv sync

# ========== SANITY CHECK ==========
echo -e "\nSanity check: verifying all local deps are importable..."

for dep in \
  "pufferlib" \
  "carbs" \
  "metta.rl.fast_gae" \
  "metta.mettagrid.mettagrid_env" \
  "metta.mettagrid.mettagrid_c" \
  "wandb_carbs"; do
  uv run python -c "import $dep; print('✅ Found {} at {}'.format('$dep', $dep.__file__))" || {
    echo "❌ Failed to import $dep"
    exit 1
  }
done

if [ -z "$IS_DOCKER" ]; then
  # ========== INSTALL METTASCOPE ==========
  echo -e "\nSetting up MettaScope..."
  bash "mettascope/install.sh"

  # ========== CHECK AWS TOKEN SETUP ==========
  echo -e "\nSetting up AWS access..."
  bash "devops/aws/setup_aws_profiles.sh"

  echo -e "\nInstalling Skypilot..."
  bash "devops/skypilot/install.sh"
  echo -e "\nRead devops/skypilot/README.md for more information on how to set up aliases for common Skypilot commands."

  echo -e "\nSetting up Observatory CLI..."
  ./devops/observatory_login.py

  echo "✅ setup_dev.sh completed successfully!"
fi

# ========== CLEAN UP REPO ==========
echo -e "\n🧹🧹🧹 cleaning up"

EXCLUDE_PATTERN="-name personal -o -name wandb -o -name train_dir"

for i in {1..5}; do # removing a child dir can make the parent empty, so loop a few times
  found_empty=0

  # Find and remove empty directories
  while IFS= read -r -d '' dir; do
    if rmdir "$dir" 2> /dev/null; then
      echo "  Removed: $dir"
      found_empty=1
    fi
  done < <(find "$PROJECT_DIR" -type d -empty -not \( $EXCLUDE_PATTERN \) -print0 2> /dev/null)

  # If no empty directories were found, we're done
  [ $found_empty -eq 0 ] && break
done
