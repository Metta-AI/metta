#!/usr/bin/env bash
set -euo pipefail

# Check if we should run
if [[ "${IS_MASTER:-false}" != "true" ]] || [[ "${ENABLE_GITHUB_STATUS:-false}" != "true" ]]; then
  exit 0
fi

# Required environment variables
: "${GITHUB_STATUS_STATE:?Missing GITHUB_STATUS_STATE}"
: "${GITHUB_STATUS_DESCRIPTION:?Missing GITHUB_STATUS_DESCRIPTION}"

# Read SkyPilot job ID from file and export it
if [ -f /tmp/.sky_tmp/sky_job_id ]; then
  export SKYPILOT_JOB_ID=$(cat /tmp/.sky_tmp/sky_job_id)
else
  export SKYPILOT_JOB_ID=""
fi

echo "[RUN] Setting GitHub status: ${GITHUB_STATUS_STATE} - ${GITHUB_STATUS_DESCRIPTION}"
uv run devops/skypilot/set_github_status.py || echo "[WARN] GitHub status update failed; continuing"
