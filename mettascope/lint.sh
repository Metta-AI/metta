#!/bin/bash
set -e

# Run from the script's directory
cd "$(dirname "$0")"

# Ensure dependencies are installed
npm install

# Run Biome lint
npm run lint
