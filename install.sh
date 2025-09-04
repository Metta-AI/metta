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
uv run python -m metta.setup.metta_cli symlink-setup setup || err "Failed to set up metta command in ~/.local/bin"

# Always ask for profile configuration during setup
echo "Setting up Metta configuration..."
if [ -n "$PROFILE" ]; then
  # Use provided profile directly by setting it as active profile
  echo "Setting active profile to: $PROFILE"
  uv run python -m metta.setup.metta_cli profile "$PROFILE" || err "Failed to set profile"
else
  if [ "$NON_INTERACTIVE" = "--non-interactive" ]; then
    # In non-interactive mode, use external profile as default (already the default in config.yaml)
    echo "Non-interactive mode: using 'external' profile (default)"
  else
    # Interactive mode: ask user to choose their profile
    echo ""
    echo "Which profile describes you?"
    echo "  1) External contributor/researcher (open source)"
    echo "  2) Softmax team member (internal)"  
    echo "  3) Custom setup (full configuration wizard)"
    echo ""
    printf "Choose an option [1-3]: "
    read -r choice
    
    case "$choice" in
      1)
        echo "Setting up external contributor profile..."
        uv run python -m metta.setup.metta_cli profile "external" || err "Failed to set profile"
        echo "✓ Configured for external contributors (W&B enabled, local storage)"
        ;;
      2)
        echo "Setting up Softmax team member profile..."
        uv run python -m metta.setup.metta_cli profile "softmax" || err "Failed to set profile"
        echo "✓ Configured for Softmax team (W&B: softmax-ai/metta-internal, S3 storage)"
        ;;
      3)
        echo "Starting custom configuration wizard..."
        uv run python -m metta.setup.metta_cli configure || err "Failed to run configuration wizard"
        ;;
      *)
        echo "Invalid choice. Defaulting to external contributor profile..."
        uv run python -m metta.setup.metta_cli profile "external" || err "Failed to set profile"
        echo "✓ Configured for external contributors (W&B enabled, local storage)"
        ;;
    esac
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
