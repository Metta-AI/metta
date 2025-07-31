#!/bin/bash

# Script installs MettaScope dependencies.

# Fail on errors:
set -e

# Determine base dir
if [[ $(basename "$PWD") != "mettascope" ]]; then
  cd mettascope
fi

# Install dependencies
pnpm install
pnpm run build

# Generate atlas
./tools/gen_atlas.py

# Copy the font sprites to dist (see https://rjwalters.github.io/glyph-atlas/)
echo "Copying font atlas files to dist"
cp ./data/fonts/font.json ./dist/font.json
cp ./data/fonts/font.png ./dist/font.png
