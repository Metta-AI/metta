#!/bin/bash
# Metta CLI bootstrap script
# This script ensures uv is installed and runs the metta CLI

set -e

# Get the real path of this script (resolving symlinks)
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null || realpath "${BASH_SOURCE[0]}" 2>/dev/null || echo "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
# Go to the repo root (parent of parent of setup)
cd "$SCRIPT_DIR/../.."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add uv to PATH for this session
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Check if .venv exists and dependencies are installed
if [ ! -d ".venv" ] || [ ! -f ".venv/bin/python" ]; then
    echo "Setting up Python environment..."
    uv sync
fi

# Run the metta CLI with uv run as a module
exec uv run python -m devops.setup.metta_cli "$@"
