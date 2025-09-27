#!/usr/bin/env bash
set -euo pipefail

# Shared function for coordinated cluster shutdown
# Usage: initiate_shutdown "reason"
initiate_shutdown() {
  local reason="${1:?Missing shutdown reason}"

  echo "$reason" > "$TERMINATION_REASON_FILE"

  # Worker nodes wait to allow master to coordinate
  if [[ "$IS_MASTER" != "true" ]]; then
    echo "[INFO] Worker node waiting 10 seconds for master to coordinate shutdown..."
    sleep 10
  fi

  echo "[INFO] Sending shutdown signal to wrapper (PID: $WRAPPER_PID)"
  kill -TERM "${WRAPPER_PID}" 2> /dev/null || true
}
