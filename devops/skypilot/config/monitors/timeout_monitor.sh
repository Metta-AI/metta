#!/usr/bin/env bash
set -euo pipefail

# Required environment variables
: "${CMD_PID:?Missing CMD_PID}"
: "${WRAPPER_PID:?Missing WRAPPER_PID}"
: "${MAX_RUNTIME_HOURS:?Missing MAX_RUNTIME_HOURS}"
: "${ACCUMULATED_RUNTIME:?Missing ACCUMULATED_RUNTIME}"
: "${ACCUMULATED_RUNTIME_FILE:?Missing ACCUMULATED_RUNTIME_FILE}"
: "${TERMINATION_REASON_FILE:?Missing TERMINATION_REASON_FILE}"
: "${START_TIME:?Missing START_TIME}"
: "${RESTART_COUNT:?Missing RESTART_COUNT}"
TEST_JOB_RESTART="${TEST_JOB_RESTART:-false}"  # Optional

TIMEOUT_CHECK_INTERVAL=${TIMEOUT_CHECK_INTERVAL:-60}

max_seconds=$(awk "BEGIN {print int(${MAX_RUNTIME_HOURS} * 3600)}")
remaining_at_start=$((max_seconds - ACCUMULATED_RUNTIME))
echo "[INFO] Timeout monitor started - max runtime: ${MAX_RUNTIME_HOURS} hours (${max_seconds} seconds)"
echo "[INFO] Already accumulated: ${ACCUMULATED_RUNTIME}s, remaining: ${remaining_at_start}s"
echo "[INFO] Checking every ${TIMEOUT_CHECK_INTERVAL} seconds"

force_restart_seconds=""
if [[ "${TEST_JOB_RESTART}" == "true" ]] && [ $RESTART_COUNT -eq 0 ]; then
  # Calculate 30% of remaining runtime
  force_restart_seconds=$(awk "BEGIN {print int(${remaining_at_start} * 0.3)}")
  echo "[INFO] Force restart test enabled - will restart after ${force_restart_seconds}s (30% of remaining ${remaining_at_start}s)"
fi

while true; do
  sleep "$TIMEOUT_CHECK_INTERVAL"

  # Check if main process is still alive
  if ! kill -0 "$CMD_PID" 2>/dev/null; then
    echo "[INFO] Timeout monitor: main process no longer running, exiting"
    break
  fi

  elapsed=$(($(date +%s) - START_TIME))
  total_runtime=$((ACCUMULATED_RUNTIME + elapsed))
  echo "$total_runtime" > "${ACCUMULATED_RUNTIME_FILE}"

  # restart test check
  if [[ -n "${force_restart_seconds:-}" ]] && [ $elapsed -ge $force_restart_seconds ]; then
    echo "[INFO] Force restart test triggered after ${elapsed} seconds"
    echo "force_restart_test" > "$TERMINATION_REASON_FILE"
    kill -TERM "${WRAPPER_PID}" 2>/dev/null || true
    break
  fi

  remaining=$((max_seconds - total_runtime))
  if [ $remaining -gt 0 ]; then
    elapsed_min=$((elapsed / 60))
    remaining_min=$((remaining / 60))
    echo "[INFO] Timeout Status: ${elapsed_min} minutes elapsed, ${remaining_min} minutes remaining (max: ${MAX_RUNTIME_HOURS}h)"
  else
    echo "[INFO] Timeout limit reached - terminating process group"
    echo "max_runtime_reached" > "$TERMINATION_REASON_FILE"
    kill -TERM "${WRAPPER_PID}" 2>/dev/null || true
    break
  fi
done

echo "[INFO] Timeout monitor exiting"
