#!/bin/bash
# Build Tribal Environment Python bindings using genny

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building Tribal Environment Python Bindings..."
echo "Working directory: $SCRIPT_DIR"

# Create bindings directory
mkdir -p bindings/generated

# Generate Python bindings using genny
echo "Generating bindings..."
nim r bindings/tribal_bindings.nim

# Build the shared library for the current platform
echo "Building shared library..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    nim c --app:lib --mm:arc --opt:speed \
        --outdir:bindings/generated \
        --out:bindings/generated/libtribal.dylib \
        bindings/tribal_bindings.nim
    LIBNAME="libtribal.dylib"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    nim c --app:lib --mm:arc --opt:speed \
        --outdir:bindings/generated \
        --out:bindings/generated/tribal.dll \
        bindings/tribal_bindings.nim
    LIBNAME="tribal.dll"
else
    nim c --app:lib --mm:arc --opt:speed \
        --outdir:bindings/generated \
        --out:bindings/generated/libtribal.so \
        bindings/tribal_bindings.nim
    LIBNAME="libtribal.so"
fi

echo "Built bindings/generated/$LIBNAME"
echo "Generated bindings/generated/tribal.py"
echo ""
echo "Python bindings are ready!"
echo ""
echo "Test with:"
echo "  cd $(dirname $0)  # Make sure you're in the tribal/ directory"
echo "  python test_tribal_bindings.py"
echo ""
echo "Or use in your Python code:"
echo "  import sys, os"
echo "  sys.path.insert(0, 'tribal/bindings/generated')  # Adjust path as needed"
echo "  import tribal"
echo "  config = tribal.defaultTribalConfig()"
echo "  env = tribal.TribalEnv(config)"