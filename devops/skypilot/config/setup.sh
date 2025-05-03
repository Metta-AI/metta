#!/bin/bash

# Install SkyPilot
pip install skypilot

# Check cloud credentials
sky check

# Create necessary directories
mkdir -p ~/.sky

# Copy example configurations if they don't exist
if [ ! -f ~/.sky/config.yaml ]; then
    cp config/example.yaml ~/.sky/config.yaml
fi

# Set up environment variables
export SKY_HOME=~/.sky
export SKY_CONFIG=~/.sky/config.yaml

echo "SkyPilot setup complete. Run 'sky check' to verify your configuration."
