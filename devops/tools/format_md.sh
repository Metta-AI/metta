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

# Only normalize markdown when formatting (not in check mode)
if [ "${CHECK_MODE:-false}" != "true" ]; then
  # Strip blank lines from markdown files, except those following:
  # - List items (bullet or dash)
  # - Closing code blocks (```)
  # - Numbered list items
  # Prettier will add back the appropriate blank lines (between headings, sections, etc.)
  echo "Normalizing markdown formatting by removing blank lines (with smart preservation)..."

  # If specific files were provided, process only those
  if [ ${#FORMAT_FILES[@]} -gt 0 ]; then
    for file in "${FORMAT_FILES[@]}"; do
      if [[ "$file" == *.md ]]; then
        python3 "$SCRIPT_DIR/normalize_markdown.py" "$file"
      fi
    done
  else
    # Otherwise, find all markdown files
    if [ -n "$EXCLUDE_PATTERN" ]; then
      find . -name "*.md" -type f -not -path "*/node_modules/*" -not -path "*/.git/*" | grep -v "$EXCLUDE_PATTERN" | while read -r file; do
        python3 "$SCRIPT_DIR/normalize_markdown.py" "$file"
      done
    else
      find . -name "*.md" -type f -not -path "*/node_modules/*" -not -path "*/.git/*" | while read -r file; do
        python3 "$SCRIPT_DIR/normalize_markdown.py" "$file"
      done
    fi
  fi

  # Add blank line before ALL CAPS emphasis patterns like **IMPORTANT**: or **NOTE**:
  echo "Adding blank lines before ALL CAPS emphasis patterns..."

  # If specific files were provided, process only those
  if [ ${#FORMAT_FILES[@]} -gt 0 ]; then
    for file in "${FORMAT_FILES[@]}"; do
      if [[ "$file" == *.md ]]; then
        sed -i '' 's/\(.\)\(\*\*[A-Z][A-Z ]*\*\*:\)/\1\
\
\2/g' "$file"
      fi
    done
  else
    # Otherwise, find all markdown files
    if [ -n "$EXCLUDE_PATTERN" ]; then
      find . -name "*.md" -type f -not -path "*/node_modules/*" -not -path "*/.git/*" | grep -v "$EXCLUDE_PATTERN" | while read -r file; do
        sed -i '' 's/\(.\)\(\*\*[A-Z][A-Z ]*\*\*:\)/\1\
\
\2/g' "$file"
      done
    else
      find . -name "*.md" -type f -not -path "*/node_modules/*" -not -path "*/.git/*" | while read -r file; do
        sed -i '' 's/\(.\)\(\*\*[A-Z][A-Z ]*\*\*:\)/\1\
\
\2/g' "$file"
      done
    fi
  fi
fi

# Format or check markdown files with Prettier
format_files "md"

echo "All Markdown files (except excluded ones) have been formatted with Prettier."
