#!/bin/sh
set -u
PROFILE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --profile)
      if [ $# -lt 2 ]; then
        echo "Error: --profile requires an argument"
        exit 1
      fi
      PROFILE="$2"
      shift 2
      ;;
    --profile=*)
      PROFILE="${1#--profile=}"
      shift
      ;;
    --help | -h)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "This script:"
      echo "  1. Installs uv and python dependencies"
      echo "  2. Configures Metta for your profile"
      echo "  3. Installs components"
      echo ""
      echo "Options:"
      echo "  --profile PROFILE      Set user profile (external, cloud, or softmax)"
      echo "                         If not specified, runs interactive configuration"
      echo "  -h, --help             Show this help message"
      echo ""
      echo "Examples:"
      echo "  $0                     # Interactive setup"
      echo "  $0 --profile=softmax   # Setup for Softmax employee"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use '$0 --help' for usage information"
      exit 1
      ;;
  esac
done

# Assumes install.sh is in root of repo
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Source common functions
. "$REPO_ROOT/devops/tools/common.sh"

echo "Welcome to Metta!"

# Ensure uv is in PATH, installed, and uv project environment associated with this repo
ensure_uv_setup

uv sync || err "Failed to install Python dependencies"
uv run python -m metta.setup.metta_cli symlink-setup || err "Failed to set up metta command in ~/.local/bin"

# Only run configuration if no config exists
if [ ! -f "$REPO_ROOT/.metta/config.yaml" ] && [ ! -f "$HOME/.metta/config.yaml" ]; then
  echo "No configuration found, running setup wizard..."
  if [ -n "$PROFILE" ]; then
    uv run python -m metta.setup.metta_cli configure --profile="$PROFILE" || err "Failed to run configuration"
  else
    uv run python -m metta.setup.metta_cli configure || err "Failed to run configuration"
  fi
else
  if [ -f "$REPO_ROOT/.metta/config.yaml" ]; then
    echo "Configuration already exists at $REPO_ROOT/.metta/config.yaml"
  else
    echo "Configuration already exists at ~/.metta/config.yaml"
  fi
fi

uv run python -m metta.setup.metta_cli install || err "Failed to install components"

echo "\nSetup complete!\n"

if ! check_cmd metta; then
  echo "To start using metta, ensure ~/.local/bin is in your PATH:"
  echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
  echo ""
  echo "Add this to your shell profile (~/.bashrc, ~/.zshrc, etc.) to make it permanent."
fi
