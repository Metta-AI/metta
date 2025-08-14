#!/usr/bin/env bash

set -e
cd /workspace/metta

echo "[SETUP] Fetching latest from origin..."
git fetch origin "$METTA_GIT_REF" || git fetch --depth=1000 origin
git checkout "$METTA_GIT_REF"
echo "[SETUP] Checked out: $(git rev-parse HEAD)"

# Note that different sets of skypilot environment variables are available in "run" vs "setup"
# see https://docs.skypilot.co/en/latest/running-jobs/environment-variables.html
