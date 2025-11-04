#!/bin/bash
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/format_script_tools.sh"

# Initialize variables
EXCLUDE_PATTERN=""

# Parse command line arguments
parse_format_args "$@"

# Ensure required tools are available
ensure_pnpm
ensure_prettier
ensure_prettier_plugin "prettier-plugin-toml" "Prettier plugin for TOML files"

# Format TOML files
format_files "toml"

if [ "${FORMAT_MODE:-write}" = "check" ]; then
  echo "All TOML files (except excluded ones) pass Prettier formatting checks."
else
  echo "All TOML files (except excluded ones) have been formatted with Prettier."
fi
