#!/usr/bin/env bash

set -e
cd /workspace/metta

git config advice.detachedHead false

echo "[SETUP] Fetching latest from origin..."
git fetch origin "$METTA_GIT_REF" || git fetch --depth=1000 origin
git checkout "$METTA_GIT_REF"
echo "[SETUP] Checked out: $(git rev-parse HEAD)"

echo "[SETUP] Preconfiguring Metta profile (softmax-docker) to avoid interactive wizard..."
uv run metta configure --profile softmax-docker --non-interactive || echo "[SETUP] Profile preconfiguration failed or skipped"

echo "[SETUP] Installing Datadog agent..."
uv run metta install datadog-agent --non-interactive || echo "[SETUP] Datadog agent installation failed or skipped"

# Note that different sets of skypilot environment variables are available in "run" vs "setup"
# see https://docs.skypilot.co/en/latest/running-jobs/environment-variables.html
