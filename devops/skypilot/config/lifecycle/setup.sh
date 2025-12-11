#!/usr/bin/env bash

set -e
REPO_DIR="/workspace/metta"

# Ensure /workspace exists and is writable
if [ ! -d /workspace ]; then
  mkdir -p /workspace 2> /dev/null || sudo mkdir -p /workspace
fi
if ! sudo chown "$(id -u)":"$(id -g)" /workspace 2> /dev/null; then
  echo "[SETUP] Failed to chown /workspace; please ensure it is writable by $(id -un)" >&2
  exit 1
fi

# Ensure repo exists (AMI doesn't include it by default)
if [ ! -d "${REPO_DIR}/.git" ]; then
  if [ -d "${REPO_DIR}" ]; then
    echo "[SETUP] Found ${REPO_DIR} without a git repo; refusing to proceed. Please clean up the directory."
    exit 1
  fi

  echo "[SETUP] Cloning ${GITHUB_REPOSITORY} into ${REPO_DIR}..."
  cd /workspace
  git clone "https://github.com/${GITHUB_REPOSITORY}.git" metta
fi

cd "${REPO_DIR}"

git config advice.detachedHead false

echo "[SETUP] Fetching latest from origin..."
git fetch origin "$METTA_GIT_REF" || git fetch --depth=1000 origin
git checkout "$METTA_GIT_REF"
echo "[SETUP] Checked out: $(git rev-parse HEAD)"

echo "[SETUP] Installing system dependencies..."
bash ./install.sh --profile softmax-docker --non-interactive

# Note that different sets of skypilot environment variables are available in "run" vs "setup"
# see https://docs.skypilot.co/en/latest/running-jobs/environment-variables.html
