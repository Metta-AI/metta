#!/bin/bash
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/format_script_tools.sh"

# Initialize variables
EXCLUDE_PATTERN=""

# Note that this script assumes that a .prettierrc is present with appropriate settings for
# regular json files and vscode's flavor of jsonc

# Parse command line arguments
parse_format_args "$@"

# Ensure required tools are available
ensure_pnpm
ensure_prettier

# Format JSON files
format_files "json"

# Also format .code-workspace files
if [ -f "metta.code-workspace" ]; then
  echo "Formatting metta.code-workspace..."
  pnpm exec prettier --write metta.code-workspace
fi

echo "All JSON files (except excluded ones) have been formatted with Prettier."
