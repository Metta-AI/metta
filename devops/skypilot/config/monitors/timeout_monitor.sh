#!/usr/bin/env bash
set -euo pipefail

# Source shared utilities
source "$(dirname "$0")/monitor_utils.sh"

# Required environment variables
: "${WRAPPER_PID:?Missing WRAPPER_PID}"
: "${MAX_RUNTIME_HOURS:?Missing MAX_RUNTIME_HOURS}"
: "${ACCUMULATED_RUNTIME:?Missing ACCUMULATED_RUNTIME}"
: "${ACCUMULATED_RUNTIME_FILE:?Missing ACCUMULATED_RUNTIME_FILE}"
: "${TERMINATION_REASON_FILE:?Missing TERMINATION_REASON_FILE}"
: "${START_TIME:?Missing START_TIME}"
: "${CLUSTER_STOP_FILE:?Missing CLUSTER_STOP_FILE}"

TIMEOUT_CHECK_INTERVAL=${TIMEOUT_CHECK_INTERVAL:-60}

max_seconds=$(awk "BEGIN {print int(${MAX_RUNTIME_HOURS} * 3600)}")
remaining_at_start=$((max_seconds - ACCUMULATED_RUNTIME))

if [ "$remaining_at_start" -le 0 ]; then
  initiate_shutdown "max_runtime_reached"
  echo "[INFO] Maximum runtime already exceeded at startup"
  exit 0
fi

echo "[INFO] Timeout monitor started!"
echo "     ↳ max runtime hours: ${MAX_RUNTIME_HOURS}"
echo "     ↳ max runtime seconds: ${max_seconds}"
echo "     ↳ accumulated runtime: ${ACCUMULATED_RUNTIME}"
echo "     ↳ remaining runtime: ${remaining_at_start}"
echo "[INFO] Checking every ${TIMEOUT_CHECK_INTERVAL} seconds"

while true; do
  if [ -s "$CLUSTER_STOP_FILE" ]; then
    echo "[INFO] Cluster stop detected, timeout monitor exiting"
    break
  fi

  sleep "$TIMEOUT_CHECK_INTERVAL"

  elapsed=$(($(date +%s) - START_TIME))
  total_runtime=$((ACCUMULATED_RUNTIME + elapsed))
  echo "$total_runtime" > "${ACCUMULATED_RUNTIME_FILE}"

  remaining=$((max_seconds - total_runtime))
  if [ $remaining -gt 0 ]; then
    elapsed_min=$((elapsed / 60))
    remaining_min=$((remaining / 60))
    echo "[INFO] Timeout Status: ${elapsed_min} minutes elapsed, ${remaining_min} minutes remaining (max: ${MAX_RUNTIME_HOURS}h)"
  else
    echo "[INFO] Maximum runtime limit reached - terminating process group"
    initiate_shutdown "max_runtime_reached"
    break
  fi
done

echo "[INFO] Timeout monitor exiting"
