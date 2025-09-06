#!/bin/bash
# Build Tribal Environment Python bindings using genny

set -e

echo "🔧 Building Tribal Environment Python Bindings..."

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
        --out:libtribal.dylib \
        bindings/tribal_bindings.nim
    LIBNAME="libtribal.dylib"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    nim c --app:lib --mm:arc --opt:speed \
        --outdir:bindings/generated \
        --out:tribal.dll \
        bindings/tribal_bindings.nim
    LIBNAME="tribal.dll"
else
    nim c --app:lib --mm:arc --opt:speed \
        --outdir:bindings/generated \
        --out:libtribal.so \
        bindings/tribal_bindings.nim
    LIBNAME="libtribal.so"
fi

echo "✅ Built bindings/generated/$LIBNAME"
echo "✅ Generated bindings/generated/tribal.py"
echo ""
echo "Python bindings are ready!"
echo ""
echo "Test with:"
echo "  source /home/relh/Code/metta/.venv/bin/activate  # if using uv"
echo "  python test_tribal_bindings.py"
echo ""
echo "Or use in your Python code:"
echo "  import sys, os"
echo "  sys.path.insert(0, 'bindings/generated')"
echo "  import tribal"
echo "  env = tribal.TribalEnv(1000)"