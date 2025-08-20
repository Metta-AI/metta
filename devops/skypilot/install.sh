#! /bin/bash -e

# temporary fix until https://github.com/skypilot-org/skypilot/pull/6698 is published on pypi
curl https://raw.githubusercontent.com/skypilot-org/skypilot/b653d6ac428ebcecfc50a809ba7d61f93cdaf12c/sky/server/common.py > .venv/lib/python3.11/site-packages/sky/server/common.py

SERVER=https://skypilot-api.softmax-research.net

echo "Logging in to Skypilot API server at $SERVER"

echo "Skypilot might ask you to copy the token. What you need to do is:
1. Copy the token in browser
2. Press Ctrl+C once
3. Paste it into the terminal
"

uv run sky api login -e "$SERVER"
