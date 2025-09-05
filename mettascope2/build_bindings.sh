#!/bin/bash
# Build Tribal Environment Python bindings using genny

set -e

echo "🔧 Building Tribal Environment Python Bindings..."

# Create bindings directory
mkdir -p bindings/generated

# Generate Python bindings using genny
echo "Generating bindings..."
nim r bindings/tribal_bindings.nim

# Build the shared library
echo "Building shared library..."
nim c --app:lib --gc:arc --opt:speed \
    --outdir:bindings/generated \
    --out:Tribal.so \
    bindings/generated/Tribal.nim

echo "✅ Built bindings/generated/Tribal.so"
echo "✅ Generated bindings/generated/Tribal.py"
echo ""
echo "Python bindings are ready!"
echo "You can now:"
echo "1. Copy Tribal.py and Tribal.so to your Python project"
echo "2. Import with: from Tribal import *"