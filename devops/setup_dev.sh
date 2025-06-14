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

# check if we're in docker
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

  # Remove the virtual environment
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
  "mettagrid.mettagrid_env" \
  "mettagrid.mettagrid_c" \
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

  echo "✅ setup_dev.sh completed successfully!"
  echo -e "Activate virtual environment with: \033[32;1msource $VENV_PATH/bin/activate\033[0m"
fi
