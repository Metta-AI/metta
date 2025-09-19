#!/usr/bin/env bash
set -euo pipefail

# Source shared utilities
source "$(dirname "$0")/monitor_utils.sh"

# Required environment variables
: "${WRAPPER_PID:?Missing WRAPPER_PID}"
: "${HEARTBEAT_FILE:?Missing HEARTBEAT_FILE}"
: "${HEARTBEAT_TIMEOUT:?Missing HEARTBEAT_TIMEOUT}"
: "${TERMINATION_REASON_FILE:?Missing TERMINATION_REASON_FILE}"
: "${CLUSTER_STOP_FILE:?Missing CLUSTER_STOP_FILE}"
: "${START_TIME:?Missing START_TIME}"

HEARTBEAT_CHECK_INTERVAL=${HEARTBEAT_CHECK_INTERVAL:-30}

echo "[INFO] Heartbeat monitor started!"
echo "     ↳ heartbeat file: ${HEARTBEAT_FILE}"
echo "     ↳ heartbeat timeout: ${HEARTBEAT_TIMEOUT} seconds"
echo "     ↳ start time: ${START_TIME}"
echo "[INFO] Checking every ${HEARTBEAT_CHECK_INTERVAL} seconds"

# Write initial heartbeat using START_TIME
mkdir -p "$(dirname "$HEARTBEAT_FILE")"
echo "$START_TIME" > "$HEARTBEAT_FILE"
echo "[INFO] Initial heartbeat written with start time: $START_TIME"

LAST_HEARTBEAT_TIME=$(stat -c %Y "$HEARTBEAT_FILE" 2> /dev/null || stat -f %m "$HEARTBEAT_FILE" 2> /dev/null)
HEARTBEAT_COUNT=0

while true; do
  if [ -s "$CLUSTER_STOP_FILE" ]; then
    echo "[INFO] Cluster stop detected, heartbeat monitor exiting"
    break
  fi

  sleep "$HEARTBEAT_CHECK_INTERVAL"

  CURRENT_TIME=$(date +%s)
  CURRENT_MTIME=$(stat -c %Y "$HEARTBEAT_FILE" 2> /dev/null || stat -f %m "$HEARTBEAT_FILE" 2> /dev/null || echo 0)

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
    echo "[ERROR] Heartbeat timeout! No heartbeat for $HEARTBEAT_TIMEOUT seconds"
    initiate_shutdown "heartbeat_timeout"
    break
  fi
done

echo "[INFO] Heartbeat monitor exiting"
