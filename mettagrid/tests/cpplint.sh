#!/usr/bin/env bash
set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Go to the project root (parent of tests directory)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Running cpplint from $PROJECT_ROOT..."

# Find all C++ files in the relevant directories
find "$PROJECT_ROOT/src/metta/mettagrid" "$PROJECT_ROOT/tests" "$PROJECT_ROOT/benchmarks" -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.hpp" \) \
  | xargs cpplint --filter=-legal,-whitespace/line_length,-readability/casting,-build/include_subdir,-whitespace/indent,-readability/inheritance,-runtime/int,-readability/todo,-build/include_what_you_use
