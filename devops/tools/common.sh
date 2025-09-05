#!/bin/bash
# Common shell functions used across various scripts

err() {
  echo "ERROR: $1" >&2
  exit 1
}

check_cmd() {
  command -v "$1" > /dev/null 2>&1
  return $?
}

_find_project_root() {
  # When sourced from install.sh, SCRIPT_DIR is already the project root
  if [ -n "${REPO_ROOT:-}" ]; then
    echo "$REPO_ROOT"
  else
    # Fallback for other scripts - use the directory of the calling script
    echo "$(cd "$(dirname "$0")/../.." && pwd)"
  fi
}

PROJECT_ROOT="$(_find_project_root)"
