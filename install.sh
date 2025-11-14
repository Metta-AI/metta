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
      echo "  1. Installs uv (if needed)"
      echo "  2. Syncs Python dependencies"
      echo "  3. Configures Metta for your profile"
      echo "  4. Installs components (including bootstrap deps)"
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

# Install uv if not present
if ! check_cmd uv; then
  echo "Installing uv..."

  # Ensure common bin directories are in PATH before installing
  for dir in "$HOME/.local/bin" "$HOME/.cargo/bin"; do
    if [ -d "$dir" ] && [ ":${PATH}:" != *":${dir}:"* ]; then
      export PATH="${dir}:${PATH}"
    fi
  done

  curl -LsSf https://astral.sh/uv/install.sh | sh

  # Source cargo env if it exists (uv installer typically installs to ~/.cargo/bin)
  if [ -f "$HOME/.cargo/env" ]; then
    . "$HOME/.cargo/env"
  fi

  # Check common locations and add to PATH if needed
  for dir in "$HOME/.local/bin" "$HOME/.cargo/bin"; do
    if [ -d "$dir" ] && [ -f "$dir/uv" ] && [ ":${PATH}:" != *":${dir}:"* ]; then
      export PATH="${dir}:${PATH}"
    fi
  done

  if ! check_cmd uv; then
    echo "Error: Failed to install uv. Please install it manually from https://github.com/astral-sh/uv"
    exit 1
  fi
  echo "uv installed successfully"
fi

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
