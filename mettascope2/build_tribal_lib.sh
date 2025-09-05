#!/bin/bash
# Build tribal environment as shared library for Python integration

set -e

echo "Building Tribal Environment Shared Library..."

# Create build directory
mkdir -p build

# Compile to shared library
nim c --app:lib --gc:arc --opt:speed \
    --outdir:build \
    --out:libtribal.so \
    src/tribal/c_api.nim

echo "✓ Built build/libtribal.so"
echo "✓ Ready for Python integration"

# Show library info
if command -v ldd &> /dev/null; then
    echo ""
    echo "Library dependencies:"
    ldd build/libtribal.so
elif command -v otool &> /dev/null; then
    echo ""
    echo "Library dependencies:"
    otool -L build/libtribal.so
fi