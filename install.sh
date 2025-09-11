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
      echo "  --profile PROFILE      Set user profile (external, cloud, or softmax)"
      echo "                         If not specified, runs interactive configuration"
      echo "  --non-interactive      Run in non-interactive mode (no prompts)"
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

err() {
  echo "ERROR: $1" >&2
  exit 1
}

check_cmd() {
  command -v "$1" > /dev/null 2>&1
  return $?
}

echo "Welcome to Metta!"

for cmd in uv bazel git g++; do
  if ! check_cmd "$cmd"; then
    echo "$cmd not found. Consider running ./devops/tools/install-system.sh"
    exit 1
  fi
done

uv sync || err "Failed to install Python dependencies"
uv run python -m metta.setup.metta_cli symlink-setup setup || err "Failed to set up metta command in ~/.local/bin"
if [ -n "$PROFILE" ]; then
  uv run python -m metta.setup.metta_cli configure --profile="$PROFILE" $NON_INTERACTIVE || err "Failed to run configuration"
else
  uv run python -m metta.setup.metta_cli configure $NON_INTERACTIVE || err "Failed to run configuration"
fi
uv run python -m metta.setup.metta_cli install $NON_INTERACTIVE || err "Failed to install components"

echo "\nSetup complete!\n"

if ! check_cmd metta; then
  echo "To start using metta, ensure ~/.local/bin is in your PATH:"
  echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
  echo ""
  echo "Add this to your shell profile (~/.bashrc, ~/.zshrc, etc.) to make it permanent."
fi
