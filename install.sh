#!/bin/sh
set -eu
PROFILE_ADDITION=""
NON_INTERACTIVE_ADDITION=""
INSTALL_CUDA_EXTRAS="0"
while [ $# -gt 0 ]; do
  case "$1" in
    --profile)
      if [ $# -lt 2 ]; then
        echo "Error: --profile requires an argument"
        exit 1
      fi
      PROFILE_ADDITION="--profile=$2"
      shift 2
      ;;
    --non-interactive)
      NON_INTERACTIVE_ADDITION="--non-interactive"
      shift
      ;;
    --with-cuda-extras)
      INSTALL_CUDA_EXTRAS="1"
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
      echo "  --with-cuda-extras     Install optional CUDA extras (flash-attn/causal-conv1d)"
      echo "  -h, --help             Show this help message"
      echo ""
      echo "Examples:"
      echo "  $0                     # Interactive setup"

      echo "  $0 --profile softmax   # Setup for Softmax employee"
      echo "  $0 --with-cuda-extras  # Also install flash-attn/causal-conv1d (Linux + CUDA)"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use '$0 --help' for usage information"
      exit 1
      ;;
  esac
done

check_cmd() {
  command -v "$1" > /dev/null 2>&1
  return $?
}

echo "Welcome to Metta!"

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
bash "$SCRIPT_DIR/devops/tools/install-system.sh"

for cmd in uv bazel git g++ nimby nim; do
  if ! check_cmd "$cmd"; then
    echo "$cmd not found. Consider running ./devops/tools/install-system.sh"
    exit 1
  fi
done

uv sync
uv run python -m metta.setup.metta_cli install $PROFILE_ADDITION $NON_INTERACTIVE_ADDITION
if [ "$INSTALL_CUDA_EXTRAS" = "1" ]; then
  if [ "$(uname -s)" = "Linux" ]; then
    uv run python scripts/install_cuda_extras.py --quiet || true
  else
    echo "CUDA extras requested but host is not Linux; skipping."
  fi
fi

echo "\nSetup complete!\n"

if ! check_cmd metta; then
  echo "To start using metta, ensure ~/.local/bin is in your PATH:"
  echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
  echo ""
  echo "Add this to your shell profile (~/.bashrc, ~/.zshrc, etc.) to make it permanent."
fi
