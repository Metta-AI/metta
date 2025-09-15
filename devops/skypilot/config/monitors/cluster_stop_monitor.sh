#!/usr/bin/env bash
set -euo pipefail

# Source the log helpers
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/log_helpers.sh"

# Required environment variables
: "${CMD_PID:?Missing CMD_PID}"
: "${WRAPPER_PID:?Missing WRAPPER_PID}"
: "${CLUSTER_STOP_FILE:?Missing CLUSTER_STOP_FILE}"
: "${TERMINATION_REASON_FILE:?Missing TERMINATION_REASON_FILE}"

CLUSTER_STOP_CHECK_INTERVAL=${CLUSTER_STOP_CHECK_INTERVAL:-15}

log_info "Cluster-stop monitor started; checking every ${CLUSTER_STOP_CHECK_INTERVAL}s"

while true; do
  sleep "$CLUSTER_STOP_CHECK_INTERVAL"

  if [ -s "$CLUSTER_STOP_FILE" ]; then
    reason="$(cat "$CLUSTER_STOP_FILE" 2> /dev/null || true)"
    log_info "Cluster stop flag detected (${reason:-no-reason}); requesting shutdown"
    echo "${reason:-cluster_stop}" > "$TERMINATION_REASON_FILE"
    kill -TERM "${WRAPPER_PID}" 2> /dev/null || true
    break
  fi

done

log_info "Cluster-stop monitor exiting"
