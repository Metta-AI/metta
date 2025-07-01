#!/bin/bash
# Metta CLI wrapper script
# This script provides backwards compatibility and convenience by using uv run

set -e

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Please run './install.sh' first."
    exit 1
fi

# Check if .venv exists (indicating install.sh has been run)
if [ ! -d ".venv" ]; then
    echo "Python environment not set up. Please run './install.sh' first."
    exit 1
fi

# Run metta using uv run
exec uv run metta "$@"
