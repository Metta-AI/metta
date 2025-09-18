#!/bin/bash

set -euo pipefail

# Navigate to the repo root
cd "$(dirname "$0")/.."

# Run install.sh with softmax profile
./install.sh --profile softmax

# Build the mettagrid extension locally with Raylib enabled for developers.
# This triggers the Bazel build backend in editable mode so changes are picked up.
export WITH_RAYLIB=1
uv pip install -e mettagrid
