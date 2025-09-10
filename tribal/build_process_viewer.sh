#!/bin/bash
set -e

echo "🔨 Building Tribal Process Viewer"

# Ensure we're in the tribal directory
if [[ ! -d "src/tribal" ]]; then
    echo "❌ Must run from tribal directory"
    exit 1
fi

# Build the process viewer
echo "🔧 Compiling process viewer..."
nim c -d:release --out:tribal_process_viewer src/tribal/tribal_process_viewer.nim

if [[ -f "tribal_process_viewer" ]]; then
    echo "✅ Process viewer built successfully: tribal_process_viewer"
    echo "🎯 Run with: ./tribal_process_viewer"
else
    echo "❌ Build failed - executable not found"
    exit 1
fi