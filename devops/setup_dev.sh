#!/bin/bash

set -euo pipefail

echo "This script has been replaced by the metta command. Setting up with default settings..."

# Navigate to the repo root
cd "$(dirname "$0")/.."

# First ensure environment is set up
if [ ! -d ".venv" ]; then
    echo "Running initial setup..."
    ./install.sh
fi

# Export PATH to use metta directly
export PATH="$(pwd)/metta/setup/installer/bin:$PATH"

# Configure as a softmax user and install all configured components
metta configure --profile=softmax && metta install
