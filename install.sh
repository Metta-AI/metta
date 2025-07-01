#!/bin/bash
# Metta initial setup script
# This script ensures uv is installed and sets up the Python environment

set -e

echo "Welcome to Metta!"
echo "This script will set up your development environment."
echo

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add uv to PATH for this session
    export PATH="$HOME/.cargo/bin:$PATH"

    echo "uv has been installed. You may need to restart your shell or run:"
    echo "  export PATH=\"\$HOME/.cargo/bin:\$PATH\""
    echo
fi

# Install Python dependencies
echo "Installing Python dependencies..."
uv sync

echo
echo "Setup complete! You can now use the 'metta' command in two ways:"
echo
echo "Option 1: Using uv run (recommended - no activation needed):"
echo "  uv run metta configure"
echo "  uv run metta install"
echo "  uv run metta status"
echo
echo "Option 2: Activate the virtual environment first:"
echo "  source .venv/bin/activate"
echo "  metta configure"
echo "  metta install"
echo "  metta status"
