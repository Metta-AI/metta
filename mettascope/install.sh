#!/bin/bash

# Script installs MettaScope dependencies.

# Fail on errors:
set -e

# Install dependencies
cd mettascope
npm install --force
npm run build

# Generate atlas
./tools/gen_atlas.py
