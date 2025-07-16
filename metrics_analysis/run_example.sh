#!/bin/bash
# Run the example analysis script using uv

echo "Running Metta Metrics Analysis Example..."
echo "========================================="

# Run with uv from the metrics_analysis directory
cd "$(dirname "$0")"
uv run python examples/basic_analysis.py "$@"
