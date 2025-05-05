#! /bin/zsh -e

# Create a new virtual environment using uv
uv venv .venv/skypilot --python=3.11 --no-project
source .venv/skypilot/bin/activate

# Install SkyPilot with all cloud providers
uv pip install skypilot==0.9.2 --prerelease=allow
