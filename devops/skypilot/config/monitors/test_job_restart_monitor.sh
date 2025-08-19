#!/usr/bin/env bash
set -euo pipefail

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
force_restart_seconds=$(awk "BEGIN {print int(${remaining_at_start} * 0.3)}")

if [ "$force_restart_seconds" -le 0 ]; then
  echo "[INFO] Job restart monitor: skipping (No time remains!)"
  exit 0
fi

echo "[INFO] Test Job Restart monitor started!"
echo "     ↳ max runtime hours: ${MAX_RUNTIME_HOURS}"
echo "     ↳ max runtime seconds: ${max_seconds}"
echo "     ↳ accumulated runtime: ${ACCUMULATED_RUNTIME}"
echo "     ↳ remaining runtime: ${remaining_at_start}"
echo "     ↳ restarting after 30% of remainder: ${force_restart_seconds}"
echo "[INFO] Checking every ${RESTART_CHECK_INTERVAL} seconds"

while true; do
  if [ -s "$CLUSTER_STOP_FILE" ]; then
    echo "[INFO] Cluster stop detected, monitor exiting"
    break
  fi

  sleep "$RESTART_CHECK_INTERVAL"

  elapsed=$(($(date +%s) - START_TIME))
  total_runtime=$((ACCUMULATED_RUNTIME + elapsed))

  remaining=$((force_restart_seconds - total_runtime))
  if [ $remaining -gt 0 ]; then
    elapsed_min=$((elapsed / 60))
    remaining_min=$((remaining / 60))
    echo "[INFO] Test Job Restart Status: ${elapsed_min} minutes elapsed, ${remaining_min} minutes remaining (max: ${MAX_RUNTIME_HOURS}h)"
  else
    echo "[INFO] Test Job Restart limit reached - terminating process group"
    echo "force_restart_test" > "$TERMINATION_REASON_FILE"
    kill -TERM "${WRAPPER_PID}" 2>/dev/null || true
    break
  fi
done

echo "[INFO] Test Job Restart monitor exiting"
