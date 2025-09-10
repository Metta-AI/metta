#!/bin/bash
set -e

echo "🔨 Running Tribal with Python Test Policy from Metta Root..."

# Default to test_move for debugging Python-Nim communication
POLICY_URI="${1:-test_move}"

echo "🎯 Using policy: $POLICY_URI"
echo "   test_move = agents should move randomly (Python controls working)"
echo "   test_noop = agents should stay still (Python sends no-op actions)"
echo ""

# Run the tribal recipe with the specified policy
echo "🚀 Launching tribal environment with Python policy control..."
uv run ./tools/run.py experiments.recipes.tribal_basic.play --args policy_uri=$POLICY_URI

echo "🧹 Tribal session ended"