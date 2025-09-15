#!/usr/bin/env bash
set -euo pipefail

# Source the log helpers
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/log_helpers.sh"

# Required environment variables
: "${WRAPPER_PID:?Missing WRAPPER_PID}"
: "${MAX_RUNTIME_HOURS:?Missing MAX_RUNTIME_HOURS}"
: "${ACCUMULATED_RUNTIME:?Missing ACCUMULATED_RUNTIME}"
: "${ACCUMULATED_RUNTIME_FILE:?Missing ACCUMULATED_RUNTIME_FILE}"
: "${TERMINATION_REASON_FILE:?Missing TERMINATION_REASON_FILE}"
: "${START_TIME:?Missing START_TIME}"
: "${CLUSTER_STOP_FILE:?Missing CLUSTER_STOP_FILE}"

TIMEOUT_CHECK_INTERVAL=${TIMEOUT_CHECK_INTERVAL:-60}

# Function to handle timeout termination
handle_timeout_termination() {
  log_info "Timeout limit reached - terminating process group"
  echo "max_runtime_reached" > "$TERMINATION_REASON_FILE"
  kill -TERM "${WRAPPER_PID}" 2> /dev/null || true
}

max_seconds=$(awk "BEGIN {print int(${MAX_RUNTIME_HOURS} * 3600)}")
remaining_at_start=$((max_seconds - ACCUMULATED_RUNTIME))

if [ "$remaining_at_start" -le 0 ]; then
  handle_timeout_termination
  log_info "Timeout monitor exiting after ensuring shutdown"
  exit 0
fi

log_info "Timeout monitor started!"
log_info "     ↳ max runtime hours: ${MAX_RUNTIME_HOURS}"
log_info "     ↳ max runtime seconds: ${max_seconds}"
log_info "     ↳ accumulated runtime: ${ACCUMULATED_RUNTIME}"
log_info "     ↳ remaining runtime: ${remaining_at_start}"
log_info "Checking every ${TIMEOUT_CHECK_INTERVAL} seconds"

while true; do
  if [ -s "$CLUSTER_STOP_FILE" ]; then
    log_info "Cluster stop detected, timeout monitor exiting"
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
    log_info "Timeout Status: ${elapsed_min} minutes elapsed, ${remaining_min} minutes remaining (max: ${MAX_RUNTIME_HOURS}h)"
  else
    handle_timeout_termination
    break
  fi
done

log_info "Timeout monitor exiting"
