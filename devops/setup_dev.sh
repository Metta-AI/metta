#!/bin/bash

set -euo pipefail

# Navigate to the repo root
cd "$(dirname "$0")/.."

# Run install.sh with softmax profile
./install.sh --profile=softmax
