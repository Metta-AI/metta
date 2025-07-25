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

# Install Playwright browsers
pnpm run playwright:install

# Generate atlas
./tools/gen_atlas.py
