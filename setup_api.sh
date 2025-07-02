#!/bin/bash
# Setup script for using Metta API

set -e

echo "Setting up Metta API environment..."

# Check if uv is available
if command -v uv &> /dev/null; then
    echo "Installing dependencies with uv..."
    uv sync --inexact
else
    echo "uv not found, using pip..."
    pip install -e .
fi

echo "Setup complete! You can now run: python run.py"
