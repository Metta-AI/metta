#!/bin/bash

# Wait for a specific termination reason to appear in the termination reason file
# Example: ./wait_for_termination.sh "nccl_test_failure" 30

set -euo pipefail

# Required environment variables
: "${TERMINATION_REASON_FILE:?Missing TERMINATION_REASON_FILE}"

# Arguments
EXPECTED_REASON="${1:-}"
MAX_WAIT="${2:-30}" # Default 30 seconds

# Validate arguments
if [ -z "$EXPECTED_REASON" ] || [ -z "$MAX_WAIT" ]; then
  echo "[ERROR] Usage: $0 <termination_reason> [max_wait_seconds]"
  echo "[ERROR] Example: $0 'nccl_test_failure' 30"
  exit 1
fi

# Check if termination file exists
if [ ! -f "$TERMINATION_REASON_FILE" ]; then
  echo "[INFO] Termination reason file not found at: $TERMINATION_REASON_FILE"
  echo "[INFO] Creating empty file..."
  mkdir -p "$(dirname "$TERMINATION_REASON_FILE")"
  touch "$TERMINATION_REASON_FILE"
fi

echo "[INFO] Waiting for termination reason: '$EXPECTED_REASON' (max ${MAX_WAIT}s)..."

# Wait loop
wait_count=0
while [ "$wait_count" -lt "$MAX_WAIT" ]; do
  # Check if the expected reason appears in the file
  if grep -q "$EXPECTED_REASON" "$TERMINATION_REASON_FILE" 2> /dev/null; then
    echo "[SUCCESS] Termination reason received: $EXPECTED_REASON"
    exit 0
  fi

  # Progress indicator every 5 seconds
  if [ $((wait_count % 5)) -eq 0 ] && [ $wait_count -gt 0 ]; then
    echo "[INFO] Still waiting... (${wait_count}s elapsed)"
  fi

  sleep 1
  ((wait_count++))
done

# Timeout reached
echo "[WARNING] Termination reason not received after ${MAX_WAIT}s"
echo "[INFO] Current termination file content:"
if [ -s "$TERMINATION_REASON_FILE" ]; then
  cat "$TERMINATION_REASON_FILE"
else
  echo "(empty)"
fi

exit 0
