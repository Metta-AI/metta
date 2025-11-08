#!/bin/bash
# Formatting-related helper functions

check_cmd() {
  command -v "$1" > /dev/null 2>&1
  return $?
}

# Check if pnpm is available, exit if not
ensure_pnpm() {
  if ! check_cmd pnpm; then
    echo "Error: pnpm not found. Please install pnpm."
    echo "You can install it using one of these methods:"
    echo "  - npm: npm install -g pnpm"
    echo "  - Homebrew: brew install pnpm"
    echo "  - Standalone: curl -fsSL https://get.pnpm.io/install.sh | sh -"
    echo "  - Or visit: https://pnpm.io/installation"
    exit 1
  fi
}

# Check if prettier is installed, offer to install if not
ensure_prettier() {
  if ! pnpm exec prettier --version &> /dev/null; then
    echo "Prettier not found. Would you like to install it? (y/n)"
    read -r answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
      echo "Installing prettier..."
      pnpm add --save-dev prettier
      echo "Prettier installed successfully."
    else
      echo "Prettier is required for this script. Exiting."
      exit 1
    fi
  fi
}

# Check if a prettier plugin is installed, offer to install if not
ensure_prettier_plugin() {
  local plugin_name="$1"
  local plugin_description="${2:-$plugin_name}"

  if ! pnpm list "$plugin_name" &> /dev/null; then
    echo "$plugin_description not found. Would you like to install it? (y/n)"
    read -r answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
      echo "Installing $plugin_name..."
      pnpm add --save-dev "$plugin_name"
      echo "$plugin_description installed successfully."
    else
      echo "$plugin_description is required for this script. Exiting."
      exit 1
    fi
  fi
}

# Function to format files with exclusion pattern
format_files() {
  local file_ext="$1"
  local exclude_pattern="${2:-$EXCLUDE_PATTERN}"

  # Determine prettier mode
  local prettier_mode="--write"
  local action="Formatting"
  if [ "${CHECK_MODE:-false}" = "true" ]; then
    prettier_mode="--check"
    action="Checking"
  fi

  # If specific files were provided via FORMAT_FILES, use those
  if [ -n "${FORMAT_FILES:-}" ]; then
    # Filter files by extension
    local matching_files=()
    for file in "${FORMAT_FILES[@]}"; do
      if [[ "$file" == *".$file_ext" ]]; then
        matching_files+=("$file")
      fi
    done

    if [ ${#matching_files[@]} -eq 0 ]; then
      echo "No *.$file_ext files in provided file list, skipping..."
      return
    fi

    pnpm exec prettier $prettier_mode "${matching_files[@]}"
  else
    # Fall back to finding all tracked files (respects .gitignore)
    if [ -n "$exclude_pattern" ]; then
      echo "  Excluding: $exclude_pattern"
      git ls-files "*.${file_ext}" | grep -v "$exclude_pattern" | xargs -r pnpm exec prettier $prettier_mode
    else
      echo "  $action all tracked files"
      git ls-files "*.${file_ext}" | xargs -r pnpm exec prettier $prettier_mode
    fi
  fi
}

# Parse common formatting script arguments
parse_format_args() {
  FORMAT_FILES=()
  while [[ $# -gt 0 ]]; do
    case $1 in
      --exclude)
        if [[ "$2" == "none" ]]; then
          EXCLUDE_PATTERN=""
        else
          EXCLUDE_PATTERN="$2"
        fi
        shift 2
        ;;
      --check)
        CHECK_MODE="true"
        shift
        ;;
      --help)
        echo "Usage: $0 [--check] [--exclude \"pattern\"] [file1 file2 ...]"
        if [ -n "${EXCLUDE_HELP_TEXT:-}" ]; then
          echo "$EXCLUDE_HELP_TEXT"
        fi
        exit 0
        ;;
      -*)
        echo "Unknown option: $1"
        echo "Usage: $0 [--check] [--exclude \"pattern\"] [file1 file2 ...]"
        if [ -n "${EXCLUDE_HELP_TEXT:-}" ]; then
          echo "$EXCLUDE_HELP_TEXT"
        fi
        exit 1
        ;;
      *)
        # Positional argument - treat as file path
        FORMAT_FILES+=("$1")
        shift
        ;;
    esac
  done
}
