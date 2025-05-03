#!/bin/bash

# Check if npx and prettier are available
if ! command -v npx &> /dev/null; then
    echo "Error: npx not found. Please install Node.js and npm."
    echo "On Mac, you can install it using:"
    echo "  brew install node"
    echo "Or download from https://nodejs.org/"
    exit 1
fi

# Check if prettier is installed
if ! npx prettier --version &> /dev/null; then
    echo "Prettier not found. Would you like to install it? (y/n)"
    read -r answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        echo "Installing prettier..."
        npm install --save-dev prettier
        echo "Prettier installed successfully."
    else
        echo "Prettier is required for this script. Exiting."
        exit 1
    fi
fi

# Format all .yml files
find . -name "*.yml" -type f | xargs npx prettier --write

# Format all .yaml files
find . -name "*.yaml" -type f | xargs npx prettier --write

echo "All YAML files have been formatted with Prettier."