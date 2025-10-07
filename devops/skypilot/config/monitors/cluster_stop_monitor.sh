#!/usr/bin/env bash
set -euo pipefail

# Required environment variables
: "${CMD_PID:?Missing CMD_PID}"
: "${WRAPPER_PID:?Missing WRAPPER_PID}"
: "${CLUSTER_STOP_FILE:?Missing CLUSTER_STOP_FILE}"
: "${TERMINATION_REASON_FILE:?Missing TERMINATION_REASON_FILE}"

CLUSTER_STOP_CHECK_INTERVAL=${CLUSTER_STOP_CHECK_INTERVAL:-15}

echo "[INFO] Cluster-stop monitor started; checking every ${CLUSTER_STOP_CHECK_INTERVAL}s"

while true; do
  sleep "$CLUSTER_STOP_CHECK_INTERVAL"

  if [ -s "$CLUSTER_STOP_FILE" ]; then
    reason="$(cat "$CLUSTER_STOP_FILE" 2> /dev/null || true)"
    # Note: Don't use monitor_utils::initiate_shutdown here since we're responding to
    # an existing cluster stop signal, not initiating one
    echo "[INFO] Cluster stop flag detected (${reason:-no-reason})"
    echo "${reason:-cluster_stop}" > "$TERMINATION_REASON_FILE"
    kill -TERM "${WRAPPER_PID}" 2> /dev/null || true
    break
  fi

  if ! kill -0 "$WRAPPER_PID" 2> /dev/null; then
    echo "[INFO] Wrapper PID $WRAPPER_PID is no longer running, exiting cluster stop monitor"
    break
  fi
done

echo "[INFO] Cluster-stop monitor exiting"
