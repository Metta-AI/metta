#!/bin/bash
# SkyPilot installation and authentication script

set -euo pipefail

# SkyPilot API server endpoint
SERVER="https://skypilot-api.softmax-research.net"

# Update SkyPilot to latest version
echo "Updating SkyPilot to latest version..."
uv run pip install -U skypilot

echo ""
echo "Attempting login to ${SERVER}..."
echo "Browser will open. If callback works, login will complete automatically."
echo ""
echo "IMPORTANT: When the browser window opens for SkyPilot login, do NOT use Safari - it fails to hand the token back. Use Chrome, Firefox, or another browser instead."
echo ""
echo "Some popup, ad, and tracker blockers (like Brave Shield) may also prevent the token from being automatically passed back to the CLI."
echo ""

# Run the actual sky api login command
uv run sky api login -e "$SERVER"
