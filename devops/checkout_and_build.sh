#!/bin/bash

# This script checks out and builds all dependencies

if [ "$SKIP_BUILD" = "1" ]; then
    echo "SKIP_BUILD was set. Skipping checkout and build!"
    exit 0
fi

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Exit immediately if a command exits with a non-zero status
set -e

cd "$SCRIPT_DIR/.."

# Call the dedicated build_mettagrid.sh script instead of building directly
echo "Building mettagrid using devops/build_mettagrid.sh"
bash "$SCRIPT_DIR/build_mettagrid.sh"

