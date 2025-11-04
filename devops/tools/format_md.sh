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

# Strip all blank lines from markdown files to normalize formatting
# Prettier will add back the appropriate blank lines (between headings, sections, etc.)
# but will NOT add blank lines in nested lists (which is what we want for consistency)
echo "Normalizing markdown formatting by removing blank lines..."
if [ -n "$EXCLUDE_PATTERN" ]; then
  find . -name "*.md" -type f -not -path "*/node_modules/*" -not -path "*/.git/*" | grep -v "$EXCLUDE_PATTERN" | while read -r file; do
    sed -i '' '/^$/d' "$file"
  done
else
  find . -name "*.md" -type f -not -path "*/node_modules/*" -not -path "*/.git/*" | while read -r file; do
    sed -i '' '/^$/d' "$file"
  done
fi

# Format markdown files with Prettier (will add back appropriate blank lines)
format_files "md"

# Also format README files without extension (if any)
echo "Checking for README files without extension..."
if [ -n "$EXCLUDE_PATTERN" ]; then
  find . -name "README" -type f | grep -v "$EXCLUDE_PATTERN" | xargs -r sed -i '' '/^$/d' | xargs -r pnpm exec prettier --write --parser markdown
else
  find . -name "README" -type f | xargs -r sed -i '' '/^$/d' | xargs -r pnpm exec prettier --write --parser markdown
fi

echo "All Markdown files (except excluded ones) have been formatted with Prettier."
