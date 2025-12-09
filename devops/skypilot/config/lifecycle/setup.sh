#!/usr/bin/env bash

set -e
cd /workspace/metta

git config advice.detachedHead false

echo "[SETUP] Fetching latest from origin..."
git fetch origin "$METTA_GIT_REF" || git fetch --depth=1000 origin
git checkout "$METTA_GIT_REF"
echo "[SETUP] Checked out: $(git rev-parse HEAD)"

# TEMP: Verify checkout succeeded and clear Python cache
# TODO: Remove this temporary fix after arch_type issue is resolved
if [ "$(git rev-parse HEAD)" != "$METTA_GIT_REF" ]; then
  echo "[SETUP] ERROR: Checkout failed! Expected $METTA_GIT_REF but got $(git rev-parse HEAD)"
  exit 1
fi

echo "[SETUP] Clearing Python bytecode cache (temporary fix)..."
find . -type d -name __pycache__ -exec rm -r {} + 2> /dev/null || true
find . -name "*.pyc" -delete 2> /dev/null || true
find . -name "*.pyo" -delete 2> /dev/null || true
# END TEMP

echo "[SETUP] Installing system dependencies..."
bash ./install.sh --profile softmax-docker --non-interactive

# TODO: reintroduce datadog agent when actually functional
# echo "[SETUP] Installing Datadog agent..."
# uv run metta install datadog-agent --non-interactive --profile=softmax-docker || echo "[SETUP] Datadog agent installation failed or skipped"

# Note that different sets of skypilot environment variables are available in "run" vs "setup"
# see https://docs.skypilot.co/en/latest/running-jobs/environment-variables.html
