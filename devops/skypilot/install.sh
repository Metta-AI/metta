#! /bin/bash -e

SERVER=https://skypilot-api.softmax-research.net

echo "Logging in to Skypilot API server at $SERVER"

uv run sky api login -e "$SERVER"
