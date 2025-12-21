#!/bin/bash
# SkyPilot installation and authentication script

set -euo pipefail

# SkyPilot API server endpoint
SERVER="https://skypilot-api.softmax-research.net"

echo ""
echo "Attempting login to ${SERVER}..."
echo "Browser will open. If callback works, login will complete automatically."

echo "IMPORTANT: When the browser window opens for SkyPilot login, do NOT use Safari - it fails to hand the token back. Use Chrome, Firefox, or another browser instead."
echo ""
echo "Some popup, ad, and tracker blockers (like Brave Shield) may also prevent the token from being automatically passed back to the CLI."
echo ""
echo "SkyPilot might ask you to copy the token. What you need to do is:"
echo "1. Copy the token in browser"
echo "2. Press Ctrl+C once"
echo "3. Paste it into the terminal"

# Run the actual sky api login command with terminal fix for token truncation
uv run python devops/skypilot/login.py api login -e "$SERVER"
