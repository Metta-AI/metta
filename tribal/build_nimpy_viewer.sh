#!/bin/bash
set -e

echo "🔨 Building Tribal Nimpy Viewer..."

cd "$(dirname "$0")"

# Compile the nimpy visualization module
nim c --app:lib --out:tribal_nimpy_viewer.so -d:release src/tribal_nimpy_viewer.nim

echo "✅ Build complete: tribal_nimpy_viewer.so"