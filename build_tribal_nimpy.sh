#!/bin/bash
set -e

echo "🔨 Building Tribal Nimpy Viewer from Metta Root..."

# Change to tribal directory for compilation
cd tribal

# Build the nimpy viewer
echo "🔧 Compiling nimpy viewer..."
nim c --app:lib --out:tribal_nimpy_viewer.so -d:release src/tribal_nimpy_viewer.nim

echo "✅ Nimpy viewer built: tribal/tribal_nimpy_viewer.so"

# Return to metta root
cd ..

echo "🎯 Nimpy viewer ready for use from metta root"