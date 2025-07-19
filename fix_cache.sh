#!/bin/bash
# Fix for cached functions.pyc files causing import errors

echo "=== Fixing cached functions.pyc files ==="
echo

# Remove the specific problematic cache files
echo "Removing cached functions.pyc files..."
rm -f metta/rl/__pycache__/functions.cpython-*.pyc
rm -f metta/rl/__pycache__/functions.cpython-*.opt-*.pyc

# Also clean any other Python cache just to be safe
echo "Cleaning all Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true

echo
echo "Verifying fix..."
python debug_imports.py

echo
echo "Testing import..."
python -c "from metta.rl.functions import cleanup_old_policies; print('✓ Import test successful!')" || echo "✗ Import still failing"

echo
echo "=== Fix complete ==="
