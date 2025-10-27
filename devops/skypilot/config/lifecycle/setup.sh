#!/usr/bin/env bash

set -e
cd /workspace/metta

git config advice.detachedHead false

echo "[SETUP] Fetching latest from origin..."
git fetch origin "$METTA_GIT_REF" || git fetch --depth=1000 origin
git checkout "$METTA_GIT_REF"
echo "[SETUP] Checked out: $(git rev-parse HEAD)"

echo "[SETUP] Installing system dependencies..."
bash ./devops/tools/install-system.sh

echo "[SETUP] Installing Datadog agent..."
uv run metta install datadog-agent --non-interactive || echo "[SETUP] Datadog agent installation failed or skipped"

echo "[SETUP] Installing Python dependencies (including Ray)..."
uv sync --frozen || echo "[SETUP] Warning: uv sync failed, dependencies may be incomplete"

# Verify Ray is installed
if ! .venv/bin/ray --version >/dev/null 2>&1; then
    echo "[SETUP] ERROR: Ray is not installed! Attempting manual installation..."
    uv pip install "ray[tune]>=2.50.1" || echo "[SETUP] Failed to install Ray"
fi

# Note that different sets of skypilot environment variables are available in "run" vs "setup"
# see https://docs.skypilot.co/en/latest/running-jobs/environment-variables.html
