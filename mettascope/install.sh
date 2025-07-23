#!/bin/bash

# Script installs MettaScope dependencies.

# Fail on errors:
set -e

# Determine base dir
if [[ $(basename "$PWD") != "mettascope" ]]; then
  cd mettascope
fi

# Install dependencies
npm install --force
npm run build

# Generate atlas
./tools/gen_atlas.py

# Copy the font sprites to dist
cp ./data/fonts/bitty.json ./dist/bitty.json
cp ./data/fonts/bitty.png ./dist/bitty.png
