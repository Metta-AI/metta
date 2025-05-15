#!/bin/bash

# Script installs MettaScope dependencies.

# Fail on errors:
set -e

# Install dependencies
cd mettascope
npm install -g typescript
npm install
tsc

# Generate atlas
python tools/gen_atlas.py
