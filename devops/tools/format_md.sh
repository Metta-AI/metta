#!/bin/bash
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/format_script_tools.sh"

# Initialize variables
EXCLUDE_PATTERN=""

# Note that this script assumes that a .prettierrc is present with appropriate settings for
# markdown files

# Parse command line arguments
parse_format_args "$@"

# Ensure required tools are available
ensure_pnpm
ensure_prettier

# Format markdown files
format_files "md"

# Also format README files without extension (if any)
echo "Checking for README files without extension..."
if [ -n "$EXCLUDE_PATTERN" ]; then
  find . -name "README" -type f | grep -v "$EXCLUDE_PATTERN" | xargs -r pnpm exec prettier --write --parser markdown
else
  find . -name "README" -type f | xargs -r pnpm exec prettier --write --parser markdown
fi

echo "All Markdown files (except excluded ones) have been formatted with Prettier."
