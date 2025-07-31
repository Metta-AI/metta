#! /bin/bash -e

SERVER=https://skypilot-api.softmax-research.net

echo "Logging in to Skypilot API server at $SERVER"

echo "Skypilot might ask you to copy the token. What you need to do is:
1. Copy the token in browser
2. Press Ctrl+C once
3. Paste it into the terminal
"

uv run sky api login -e "$SERVER"
