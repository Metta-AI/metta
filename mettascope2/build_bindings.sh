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
nim c --app:lib --mm:arc --opt:speed \
    --outdir:bindings/generated \
    --out:libtribal.dylib \
    bindings/tribal_bindings.nim

# Fix the generated Python file to use correct module name
echo "Fixing generated Python file..."
sed -i '' 's/sys.modules\["tribal"\]/sys.modules["Tribal"]/g' bindings/generated/Tribal.py

echo "✅ Built bindings/generated/libtribal.dylib"
echo "✅ Generated and fixed bindings/generated/Tribal.py"
echo ""
echo "Python bindings are ready!"
echo "You can now:"
echo "1. Copy Tribal.py and libtribal.dylib to your Python project"
echo "2. Import with: from Tribal import *"