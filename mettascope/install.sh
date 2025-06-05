#!/bin/bash

# Script installs MettaScope dependencies.

# Fail on errors:
set -e

# Install dependencies
cd mettascope
npm install --force -g typescript
npm install --force
tsc

# Generate atlas
./tools/gen_atlas.py
