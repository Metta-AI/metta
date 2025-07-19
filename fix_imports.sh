#!/bin/bash
# Script to fix metta.rl.functions import issues

echo "=== Fixing metta.rl.functions import issues ==="
echo

# Check current branch
echo "Current branch:"
git branch --show-current
echo

# Pull latest changes
echo "Pulling latest changes..."
git pull origin richard-func-to-folder
echo

# Clean up any leftover functions.py file
if [ -f "metta/rl/functions.py" ]; then
    echo "Removing leftover functions.py file..."
    rm -f metta/rl/functions.py
fi

# Clean up Python cache
echo "Cleaning Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
echo

# Verify the structure
echo "Verifying structure..."
if [ -d "metta/rl/functions" ] && [ -f "metta/rl/functions/__init__.py" ]; then
    echo "✓ functions/ directory exists with __init__.py"
else
    echo "✗ Missing functions/ directory or __init__.py"
    exit 1
fi

if [ -f "metta/rl/functions.py" ]; then
    echo "✗ functions.py file still exists!"
    exit 1
else
    echo "✓ No functions.py file (correct)"
fi

echo
echo "Testing import..."
python -c "from metta.rl.functions import cleanup_old_policies; print('✓ Import successful')" || echo "✗ Import failed"

echo
echo "=== Fix complete ==="
