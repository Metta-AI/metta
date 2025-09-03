#!/usr/bin/env bash
set -euo pipefail

# Check if we should run
if [[ "${IS_MASTER:-false}" != "true" ]] || [[ "${ENABLE_DISCORD:-false}" != "true" ]]; then
  exit 0
fi

# Required arguments
EMOJI="${1:?Missing emoji argument}"
TITLE="${2:?Missing title argument}"
STATUS_MSG="${3:?Missing status message argument}"
ADDITIONAL_INFO="${4:-}"

# Required environment variables
: "${GITHUB_REPOSITORY:?Missing GITHUB_REPOSITORY}"
: "${METTA_GIT_REF:?Missing METTA_GIT_REF}"
: "${METTA_RUN_ID:?Missing METTA_RUN_ID}"
: "${TOTAL_NODES:?Missing TOTAL_NODES}"
: "${JOB_METADATA_DIR:?Missing JOB_METADATA_DIR}"

echo "[RUN] Sending Discord notification: $TITLE"

# Calculate runtime if START_TIME is set
runtime_msg=""
if [ -n "${START_TIME:-}" ] && [ "${START_TIME}" -ne 0 ]; then
  current_time=$(date +%s)
  duration=$((current_time - START_TIME))
  hours=$((duration / 3600))
  minutes=$(((duration % 3600) / 60))
  runtime_msg="**Runtime**: ${hours}h ${minutes}m"
fi

# Build Discord message
{
  echo "$EMOJI **$TITLE**"
  echo ""
  echo "**Repository**: ${GITHUB_REPOSITORY}"
  echo "**Git Ref**: ${METTA_GIT_REF}"
  echo "**Run ID**: ${METTA_RUN_ID:-N/A}"
  echo "**Status**: $STATUS_MSG"
  [ -n "$runtime_msg" ] && echo "$runtime_msg"
  echo "**Time**: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
  echo "**Nodes**: ${TOTAL_NODES}"
  [ -n "$ADDITIONAL_INFO" ] && echo "" && echo "$ADDITIONAL_INFO"
} > "$JOB_METADATA_DIR/discord_message.txt"

DISCORD_CONTENT="$(cat "$JOB_METADATA_DIR/discord_message.txt")"
export DISCORD_CONTENT
uv run -m metta.common.util.discord || echo "[WARN] Discord notification failed; continuing"
