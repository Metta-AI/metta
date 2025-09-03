#!/bin/sh
set -u
PROFILE=""
NON_INTERACTIVE=""
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
    --non-interactive)
      NON_INTERACTIVE="--non-interactive"
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
      echo "  --profile PROFILE      Set user profile (external, softmax)"
      echo "                         If not specified, shows interactive menu"
      echo "  --non-interactive      Run in non-interactive mode (uses 'external' defaults)"
      echo "  -h, --help             Show this help message"
      echo ""
      echo "Examples:"
      echo "  $0                           # Interactive menu: choose defaults or full wizard"
      echo "  $0 --profile=softmax         # Quick setup for Softmax employees"
      echo "  $0 --profile=external        # Quick setup for external users"
      echo "  $0 --non-interactive         # Automated setup with external defaults"
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
    # Use provided profile directly
    uv run python -m metta.setup.metta_cli configure --profile="$PROFILE" $NON_INTERACTIVE || err "Failed to run configuration"
  else
    if [ "$NON_INTERACTIVE" = "--non-interactive" ]; then
      # In non-interactive mode, use external profile as default
      echo "Non-interactive mode: using 'external' profile defaults"
      uv run python -m metta.setup.metta_cli configure --profile="external" $NON_INTERACTIVE || err "Failed to run configuration"
    else
      # Interactive mode: ask user what they want to do
      echo ""
      echo "Configuration options:"
      echo "  1) Quick setup with defaults for external users"
      echo "  2) Quick setup with defaults for Softmax employees"  
      echo "  3) Full configuration wizard (customize everything)"
      echo ""
      printf "Choose an option [1-3]: "
      read -r choice
      
      case "$choice" in
        1)
          echo "Using external user defaults..."
          uv run python -m metta.setup.metta_cli configure --profile="external" $NON_INTERACTIVE || err "Failed to run configuration"
          ;;
        2)
          echo "Using Softmax employee defaults..."
          uv run python -m metta.setup.metta_cli configure --profile="softmax" $NON_INTERACTIVE || err "Failed to run configuration"
          ;;
        3)
          echo "Starting full configuration wizard..."
          uv run python -m metta.setup.metta_cli configure $NON_INTERACTIVE || err "Failed to run configuration"
          ;;
        *)
          echo "Invalid choice. Using external user defaults..."
          uv run python -m metta.setup.metta_cli configure --profile="external" $NON_INTERACTIVE || err "Failed to run configuration"
          ;;
      esac
    fi
  fi
else
  if [ -f "$REPO_ROOT/.metta/config.yaml" ]; then
    echo "Configuration already exists at $REPO_ROOT/.metta/config.yaml"
  else
    echo "Configuration already exists at ~/.metta/config.yaml"
  fi
fi

uv run python -m metta.setup.metta_cli install $NON_INTERACTIVE || err "Failed to install components"

echo "\nSetup complete!\n"

if ! check_cmd metta; then
  echo "To start using metta, ensure ~/.local/bin is in your PATH:"
  echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
  echo ""
  echo "Add this to your shell profile (~/.bashrc, ~/.zshrc, etc.) to make it permanent."
fi
