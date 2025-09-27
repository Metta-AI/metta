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
: "${RESTART_COUNT:?Missing RESTART_COUNT}"

RESTART_CHECK_INTERVAL=${RESTART_CHECK_INTERVAL:-30}

# Only run on first attempt
if [ "$RESTART_COUNT" -ne 0 ]; then
  echo "[INFO] Job restart monitor: skipping (RESTART_COUNT=$RESTART_COUNT)"
  exit 0
fi

max_seconds=$(awk "BEGIN {print int(${MAX_RUNTIME_HOURS} * 3600)}")
remaining_at_start=$((max_seconds - ACCUMULATED_RUNTIME))
force_restart_delay=$(awk "BEGIN {print int(${remaining_at_start} * 0.5)}")

if [ "$force_restart_delay" -le 0 ]; then
  echo "[INFO] Job restart monitor: skipping (No time remains!)"
  exit 0
fi

echo "[INFO] Test Job Restart monitor started!"
echo "     ↳ restarting after: ${force_restart_delay}"
echo "[INFO] Checking every ${RESTART_CHECK_INTERVAL} seconds"

while true; do
  if [ -s "$CLUSTER_STOP_FILE" ]; then
    echo "[INFO] Cluster stop detected, test job restart monitor exiting"
    break
  fi

  sleep "$RESTART_CHECK_INTERVAL"

  elapsed=$(($(date +%s) - START_TIME))

  remaining=$((force_restart_delay - elapsed))
  if [ $remaining -gt 0 ]; then
    elapsed_min=$((elapsed / 60))
    remaining_min=$((remaining / 60))
    echo "[INFO] Test Job Restart Status: ${elapsed_min} minutes elapsed, ${remaining_min} minutes remaining until job restart test"
  else
    echo "[INFO] Test job restart limit reached - terminating process group"
    initiate_shutdown "force_restart_test"
    break
  fi
done

echo "[INFO] Test Job Restart monitor exiting"
