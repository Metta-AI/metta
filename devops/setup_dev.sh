#!/bin/bash

set -euo pipefail

echo "This script has been replaced by metta.sh. Will run it with some default settings..."

# Navigate to the repo root
cd "$(dirname "$0")/.."

# Configure as a softmax user and install all configured components
./metta.sh configure --profile=softmax && ./metta.sh install
