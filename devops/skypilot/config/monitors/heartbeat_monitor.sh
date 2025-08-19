#!/usr/bin/env bash
set -euo pipefail

# Required environment variables
: "${WRAPPER_PID:?Missing WRAPPER_PID}"
: "${HEARTBEAT_FILE:?Missing HEARTBEAT_FILE}"
: "${HEARTBEAT_TIMEOUT:?Missing HEARTBEAT_TIMEOUT}"
: "${TERMINATION_REASON_FILE:?Missing TERMINATION_REASON_FILE}"
: "${CLUSTER_STOP_FILE:?Missing CLUSTER_STOP_FILE}"
: "${START_TIME:?Missing START_TIME}"

HEARTBEAT_CHECK_INTERVAL=${HEARTBEAT_CHECK_INTERVAL:-30}

echo "[INFO] Heartbeat monitor started - timeout: ${HEARTBEAT_TIMEOUT}s, file: ${HEARTBEAT_FILE}"
echo "[INFO] Checking every ${HEARTBEAT_CHECK_INTERVAL} seconds"

LAST_HEARTBEAT_TIME=$(date +%s)
HEARTBEAT_COUNT=0

stop_cluster() {
  local msg="$1"
  echo "[ERROR] Heartbeat timeout! $msg"
  echo "heartbeat_timeout" > "$TERMINATION_REASON_FILE"
  kill -TERM "${WRAPPER_PID}" 2>/dev/null || true
}

while true; do
  if [ -s "$CLUSTER_STOP_FILE" ]; then
    echo "[INFO] Cluster stop detected, heartbeat monitor exiting"
    break
  fi

  sleep "$HEARTBEAT_CHECK_INTERVAL"

  CURRENT_TIME=$(date +%s)

  if [ -f "$HEARTBEAT_FILE" ]; then
    CURRENT_MTIME=$(stat -c %Y "$HEARTBEAT_FILE" 2>/dev/null || stat -f %m "$HEARTBEAT_FILE" 2>/dev/null || echo 0)

    if [ "$CURRENT_MTIME" -gt "$LAST_HEARTBEAT_TIME" ]; then
      HEARTBEAT_COUNT=$((HEARTBEAT_COUNT + 1))
      LAST_HEARTBEAT_TIME=$CURRENT_MTIME

      # Print status occasionally
      if [ $((HEARTBEAT_COUNT % 10)) -eq 0 ]; then
        echo "[INFO] Heartbeat received! (Total: $HEARTBEAT_COUNT heartbeat checks)"
      fi
    fi

    # Check if timeout exceeded
    if [ $((CURRENT_TIME - LAST_HEARTBEAT_TIME)) -gt "$HEARTBEAT_TIMEOUT" ]; then
      stop_cluster "No heartbeat for $HEARTBEAT_TIMEOUT seconds"
      break
    fi
  else
    # If the heartbeat file never appeared, enforce timeout from start
    if [ $((CURRENT_TIME - START_TIME)) -gt "$HEARTBEAT_TIMEOUT" ]; then
      stop_cluster "Heartbeat file never appeared in $HEARTBEAT_TIMEOUT seconds"
      break
    fi
  fi

done

echo "[INFO] Heartbeat monitor exiting"
