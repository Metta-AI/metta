#!/usr/bin/env bash
set -euo pipefail

echo "Running cpplint..."
find mettagrid/mettagrid mettagrid/tests mettagrid/benchmarks -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.hpp" \) \
    -not -path "*/third_party/*" \
    | xargs cpplint --filter=-legal,-whitespace/line_length,-readability/casting,-build/include_subdir,-whitespace/indent,-readability/inheritance,-runtime/int,-readability/todo,-build/include_what_you_use
