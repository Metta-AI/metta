#!/bin/bash

# Script installs MettaScope dependencies.

# Fail on errors:
set -e

# Install npm
brew install npm

# Install dependencies
cd mettascope
npm install
tsc

# Generate atlas
python tools/gen_atlas.py
