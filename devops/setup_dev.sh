#!/bin/bash

set -euo pipefail

echo "This script has been replaced by the metta command. Setting up with default settings..."

# Navigate to the repo root
cd "$(dirname "$0")/.."

# Run installer with --add-to-path flag for non-interactive setup
./install.sh --add-to-path

# Use the metta wrapper directly with full path (since PATH changes won't take effect until new shell)
./metta/setup/installer/bin/metta configure --profile=softmax && ./metta/setup/installer/bin/metta install
