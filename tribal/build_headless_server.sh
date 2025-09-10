#!/bin/bash
set -e

echo "🔨 Building Tribal Headless Server"

# Ensure we're in the tribal directory
if [[ ! -d "src/tribal" ]]; then
    echo "❌ Must run from tribal directory"
    exit 1
fi

# Build the headless server
echo "🔧 Compiling headless server..."
nim c -d:release --out:tribal_headless_server src/tribal_headless_server.nim

if [[ -f "tribal_headless_server" ]]; then
    echo "✅ Headless server built successfully: tribal_headless_server"
    echo "🎯 Run with: ./tribal_headless_server"
else
    echo "❌ Build failed - executable not found"
    exit 1
fi