#!/bin/bash
# Metta CLI wrapper script
# This script provides backwards compatibility and convenience

set -e

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Get the project root (two levels up from metta/setup/)
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Use the main metta wrapper script
exec "$PROJECT_DIR/metta/setup/installer/bin/metta" "$@"
