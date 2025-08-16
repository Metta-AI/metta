#!/usr/bin/env bash

set -euo pipefail

cd /workspace/metta

# Drop any preloaded venv; activate your own
if [ -n "${VIRTUAL_ENV:-}" ]; then
  deactivate 2>/dev/null || true
fi
. .venv/bin/activate

# Determine node role using SkyPilot environment variables
RANK=${SKYPILOT_NODE_RANK:-0}
export IS_MASTER=$([[ "$RANK" == "0" ]] && echo "true" || echo "false")
TOTAL_NODES=${SKYPILOT_NUM_NODES:-1}

# Master-only: Collect SkyPilot latency
if [[ "$IS_MASTER" == "true" ]]; then
  if [ -f common/src/metta/common/util/skypilot_latency.py ]; then
    echo "[RUN] Collecting skypilot latency..."
    uv run python common/src/metta/common/util/skypilot_latency.py || true
  else
    echo "[RUN] Latency script is missing!"
  fi
fi

METTA_ENV_FILE="$(uv run ./common/src/metta/common/util/constants.py METTA_ENV_FILE)"

# Master-only: Collect instance cost
if [[ "$IS_MASTER" == "true" ]]; then
  if [ -f common/src/metta/common/util/cost_monitor.py ]; then
    echo "[RUN] Collecting instance cost..."
    METTA_HOURLY_COST="$(uv run python common/src/metta/common/util/cost_monitor.py 2>/dev/null | tail -1 || true)"
    if [ -n "${METTA_HOURLY_COST:-}" ]; then
      echo "[RUN] METTA_HOURLY_COST set to: $METTA_HOURLY_COST in $METTA_ENV_FILE"
    else
      echo "[RUN] Cost monitor script failed to run or returned no value."
    fi
  else
    echo "[RUN] Cost monitor script is missing!"
  fi
fi

# Setup environment (all nodes)
bash ./devops/skypilot/config/configure_environment.sh
source "$METTA_ENV_FILE"

EXIT_SUCCESS=0
EXIT_FAILURE=1
EXIT_NCCL_TEST_FAILURE=42

echo "[CONFIG] Run Configuration:"
echo "  - NODE_RANK: ${RANK}"
echo "  - IS_MASTER: ${IS_MASTER}"
echo "  - TOTAL_NODES: ${TOTAL_NODES}"
echo "  - METTA_RUN_ID: ${METTA_RUN_ID:-}"
echo "  - SKYPILOT_TASK_ID: ${SKYPILOT_TASK_ID:-}"
echo "  - HEARTBEAT_TIMEOUT: ${HEARTBEAT_TIMEOUT:-'NOT SET'}"
echo "  - MAX_RUNTIME_HOURS: ${MAX_RUNTIME_HOURS:-'NOT SET'}"
echo "  - RESTART_COUNT: ${RESTART_COUNT}"
echo "  - ACCUMULATED_RUNTIME: ${ACCUMULATED_RUNTIME}s ($((ACCUMULATED_RUNTIME / 60))m)"
echo "  - TEST_JOB_RESTART: ${TEST_JOB_RESTART:-0}"
echo "  - METTA_CMD: ${METTA_CMD:-'NOT SET'}"
echo "  - METTA_CMD_ARGS: ${METTA_CMD_ARGS:-'NOT SET'}"

# Create a temp directory for IPC files
export IPC_DIR="/tmp/metta_job_$$"
mkdir -p "$IPC_DIR"
export TERMINATION_REASON_FILE="$IPC_DIR/termination_reason"
export CMD_PID=""
export CMD_PGID=""
export START_TIME=0

export HEARTBEAT_FILE=${HEARTBEAT_FILE:-$WANDB_DIR/heartbeat.txt}
export TERMINATION_REASON=""

# Configurable intervals
export TIMEOUT_CHECK_INTERVAL=${TIMEOUT_CHECK_INTERVAL:-60}
export HEARTBEAT_CHECK_INTERVAL=${HEARTBEAT_CHECK_INTERVAL:-30}

# Flag to prevent multiple shutdowns
export SHUTDOWN_IN_PROGRESS=0

# Set up feature flags based on available credentials (master only)
if [[ "$IS_MASTER" == "true" ]]; then
  if [ -n "${DISCORD_WEBHOOK_URL:-}" ]; then
    export ENABLE_DISCORD=true
    echo "[RUN] Discord notifications are enabled"
  else
    export ENABLE_DISCORD=false
    echo "[RUN] Discord notifications are disabled (no webhook URL)"
  fi

  if [ -n "${GITHUB_PAT:-}" ] && [ -n "${GITHUB_REPOSITORY:-}" ] && [ -n "${METTA_GIT_REF:-}" ]; then
    export ENABLE_GITHUB_STATUS=true
    echo "[RUN] GitHub status reporting is enabled"
  else
    export ENABLE_GITHUB_STATUS=false
    echo "[RUN] GitHub status reporting is disabled (missing required credentials)"
  fi
else
  export ENABLE_DISCORD=false
  export ENABLE_GITHUB_STATUS=false
fi

# Function to conditionally send Discord notifications (master only)
maybe_send_discord_notification() {
  local emoji="$1"
  local title="$2"
  local status_msg="$3"
  local additional_info="${4:-}"

  if [[ "$IS_MASTER" != "true" ]] || [ "$ENABLE_DISCORD" != "true" ]; then
    return 0
  fi

  echo "[RUN] Sending Discord notification: $title"

  # Calculate runtime if START_TIME is set
  local runtime_msg=""
  if [ -n "${START_TIME:-}" ] && [ "${START_TIME}" -ne 0 ]; then
    local current_time=$(date +%s)
    local duration=$((current_time - START_TIME))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    runtime_msg="**Runtime**: ${hours}h ${minutes}m"
  fi

  # Build Discord message
  {
    echo "$emoji **$title**"
    echo ""
    echo "**Repository**: ${GITHUB_REPOSITORY}"
    echo "**Git Ref**: ${METTA_GIT_REF}"
    echo "**Run ID**: ${METTA_RUN_ID:-N/A}"
    echo "**Status**: $status_msg"
    [ -n "$runtime_msg" ] && echo "$runtime_msg"
    echo "**Time**: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
    echo "**Nodes**: ${TOTAL_NODES}"
    [ -n "$additional_info" ] && echo "" && echo "$additional_info"
  } > "$IPC_DIR/discord_message.txt"

  DISCORD_CONTENT="$(cat "$IPC_DIR/discord_message.txt")"
  export DISCORD_CONTENT
  uv run -m metta.common.util.discord || echo "[WARN] Discord notification failed; continuing"
}

# Function to conditionally set GitHub status (master only)
maybe_set_github_status() {
  if [[ "$IS_MASTER" != "true" ]] || [ "$ENABLE_GITHUB_STATUS" != "true" ]; then
    return 0
  fi

  # Read SkyPilot job ID from file and export it
  if [ -f /tmp/.sky_tmp/sky_job_id ]; then
    export SKYPILOT_JOB_ID=$(cat /tmp/.sky_tmp/sky_job_id)
  else
    export SKYPILOT_JOB_ID=""
  fi

  echo "[RUN] Setting GitHub status: ${GITHUB_STATUS_STATE:-} - ${GITHUB_STATUS_DESCRIPTION:-}"
  uv run devops/skypilot/set_github_status.py || echo "[WARN] GitHub status update failed; continuing"
}

# Export the functions so they're available to sub-shells
export -f maybe_send_discord_notification
export -f maybe_set_github_status

# Master-only: Initial GitHub status
if [[ "$IS_MASTER" == "true" ]]; then
  export GITHUB_STATUS_STATE=pending
  export GITHUB_STATUS_DESCRIPTION="Queued on SkyPilot…"
  maybe_set_github_status
fi

graceful_shutdown() {
  # Prevent multiple simultaneous shutdowns
  if [ $SHUTDOWN_IN_PROGRESS -eq 1 ]; then
    echo "[SHUTDOWN] Shutdown already in progress, ignoring signal"
    return
  fi
  SHUTDOWN_IN_PROGRESS=1

  echo "[SHUTDOWN] Caught INT/TERM/HUP; initiating graceful shutdown..."

  # Disable the trap to prevent re-entry
  trap '' INT TERM HUP

  # If a monitor set a reason, keep it; otherwise set a generic one
  if [ -z "${TERMINATION_REASON:-}" ] && [ -f "$TERMINATION_REASON_FILE" ]; then
    TERMINATION_REASON="$(cat "$TERMINATION_REASON_FILE" || true)"
  fi
  TERMINATION_REASON="${TERMINATION_REASON:-controlled_shutdown}"

  # Kill the entire process tree forcefully
  if [ -n "${CMD_PGID:-}" ] && [ -n "${CMD_PID:-}" ]; then
    echo "[SHUTDOWN] Killing training process tree (PGID: ${CMD_PGID})"

    # First try SIGTERM to the process group
    kill -TERM -"${CMD_PGID}" 2>/dev/null || true

    # Give it 5 seconds to clean up
    local count=0
    while kill -0 "$CMD_PID" 2>/dev/null && [ $count -lt 5 ]; do
      sleep 1
      ((count++))
    done

    # If still alive, use SIGKILL
    if kill -0 "$CMD_PID" 2>/dev/null; then
      echo "[SHUTDOWN] Process didn't terminate, using SIGKILL"
      kill -KILL -"${CMD_PGID}" 2>/dev/null || true
    fi

    # Wait for the main process to die
    wait "${CMD_PID}" 2>/dev/null || true
  fi

  terminate_monitors

  # Compute duration (best-effort)
  if [ -n "${START_TIME:-}" ] && [ "${START_TIME}" -ne 0 ]; then
    local end_time
    end_time=$(date +%s)
    local dur=$((end_time - START_TIME))
    echo "[SUMMARY] Total runtime: ${dur} seconds ($((dur/60)) minutes)"
  fi

  # Report success on controlled shutdown
  CMD_EXIT=0
  FINAL_EXIT_CODE=0
  echo "[INFO] Termination reason: ${TERMINATION_REASON}"
  echo "[INFO] Exiting wrapper with code 0 (controlled shutdown)"

  export CMD_EXIT
  maybe_set_github_status

  exit $EXIT_SUCCESS
}

# Trap signals on the parent (the process SkyPilot watches)
trap graceful_shutdown INT TERM HUP

terminate_monitors() {
  if [[ -n "${HEARTBEAT_MONITOR_PID:-}" ]] && kill -0 "$HEARTBEAT_MONITOR_PID" 2>/dev/null; then
    kill "$HEARTBEAT_MONITOR_PID" 2>/dev/null || true
    wait "$HEARTBEAT_MONITOR_PID" 2>/dev/null || true
    echo "[INFO] Terminated heartbeat monitor"
  fi

  if [[ -n "${TIMEOUT_MONITOR_PID:-}" ]] && kill -0 "$TIMEOUT_MONITOR_PID" 2>/dev/null; then
    kill "$TIMEOUT_MONITOR_PID" 2>/dev/null || true
    wait "$TIMEOUT_MONITOR_PID" 2>/dev/null || true
    echo "[INFO] Terminated timeout monitor"
  fi
}

terminate_process() {
  local pid=$1
  local reason=$2

  echo "[INFO] Requesting graceful shutdown (reason: $reason)"
  echo "$reason" > "$TERMINATION_REASON_FILE"
  TERMINATION_REASON="$reason"

  # Master broadcasts the stop to all nodes via shared flag
  if [[ "$IS_MASTER" == "true" ]] && [ -n "${CLUSTER_STOP_FILE:-}" ]; then
    echo "$reason" > "$CLUSTER_STOP_FILE"
  fi

  # Only send signal if not already shutting down
  if [ $SHUTDOWN_IN_PROGRESS -eq 0 ]; then
    # Send TERM to self, which will trigger graceful_shutdown
    kill -TERM $$ 2>/dev/null || true
  fi
}

run_cmd() {
  echo "[INFO] Starting process $METTA_CMD (node rank: $RANK)"

  START_TIME=$(date +%s)

  # Start training in its own process group; tee output for postmortem
  setsid ./devops/"${METTA_CMD:?missing METTA_CMD}".sh run="${METTA_RUN_ID:?missing METTA_RUN_ID}" ${METTA_CMD_ARGS:-} 2>&1 | tee "$IPC_DIR/${METTA_CMD}_log.txt" &
  CMD_PID=$!

  sleep 1
  if ! kill -0 "$CMD_PID" 2>/dev/null; then
    echo "[ERROR] Command process died immediately!"
    return 1
  fi

  CMD_PGID=$(ps -o pgid= -p "$CMD_PID" 2>/dev/null | tr -d ' ')
  echo "[INFO] Started $METTA_CMD process with PID: $CMD_PID, PGID: $CMD_PGID"

  # Master-only: Start timeout monitor if MAX_RUNTIME_HOURS is set
  if [[ "$IS_MASTER" == "true" ]] && [[ -n "${MAX_RUNTIME_HOURS:-}" ]] && [[ "${MAX_RUNTIME_HOURS}" != "None" ]]; then
    (
      exec 2>&1
      max_seconds=$(awk "BEGIN {print int(${MAX_RUNTIME_HOURS} * 3600)}")
      remaining_at_start=$((max_seconds - ACCUMULATED_RUNTIME))
      echo "[INFO] Timeout monitor started - max runtime: ${MAX_RUNTIME_HOURS} hours (${max_seconds} seconds)"
      echo "[INFO] Already accumulated: ${ACCUMULATED_RUNTIME}s, remaining: ${remaining_at_start}s"
      echo "[INFO] Checking every ${TIMEOUT_CHECK_INTERVAL} seconds"

      force_restart_seconds=""
      if [[ -n "${TEST_JOB_RESTART:-}" ]] && [[ "${TEST_JOB_RESTART}" != "0" ]] && [ $RESTART_COUNT -eq 0 ]; then
        # Calculate 30% of remaining runtime (not total runtime)
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
          total_runtime=$((ACCUMULATED_RUNTIME + elapsed))
          echo "$total_runtime" > "${ACCUMULATED_RUNTIME_FILE}"
          terminate_process "$CMD_PID" "force_restart_test"
          break
        fi

        remaining=$((max_seconds - total_runtime))
        if [ $remaining -gt 0 ]; then
          elapsed_min=$((elapsed / 60))
          remaining_min=$((remaining / 60))
          echo "[INFO] Timeout Status: ${elapsed_min} minutes elapsed, ${remaining_min} minutes remaining (max: ${MAX_RUNTIME_HOURS}h)"
        else
          echo "[INFO] Timeout limit reached - terminating process group"
          terminate_process "$CMD_PID" "max_runtime_reached"
          break
        fi
      done
    ) &
    TIMEOUT_MONITOR_PID=$!
    echo "[INFO] Started timeout monitor with PID: $TIMEOUT_MONITOR_PID"
  fi

  if [[ "${HEARTBEAT_TIMEOUT}" != "0" ]]; then
    echo "[INFO] Starting heartbeat monitor ${HEARTBEAT_TIMEOUT}s on $HEARTBEAT_FILE"
    echo "[INFO] Checking every ${HEARTBEAT_CHECK_INTERVAL} seconds"

    (
      LAST_HEARTBEAT_TIME=$(date +%s)
      HEARTBEAT_COUNT=0

      while kill -0 "$CMD_PID" 2>/dev/null; do
        if [ -f "$HEARTBEAT_FILE" ]; then
          CURRENT_MTIME=$(stat -c %Y "$HEARTBEAT_FILE" 2>/dev/null || stat -f %m "$HEARTBEAT_FILE" 2>/dev/null || echo 0)
          CURRENT_TIME=$(date +%s)

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
            terminate_process "$CMD_PID" "heartbeat_timeout"
            break
          fi
        fi

        # --- cluster stop check ---
        if [ -n "${CLUSTER_STOP_FILE:-}" ] && [ -s "$CLUSTER_STOP_FILE" ]; then
          reason="$(cat "$CLUSTER_STOP_FILE" 2>/dev/null || true)"
          echo "[INFO] Cluster stop flag detected (${reason:-no-reason}); requesting shutdown"
          terminate_process "$CMD_PID" "${reason:-cluster_stop}"
          break
        fi

        sleep "$HEARTBEAT_CHECK_INTERVAL"
      done

      echo "[INFO] Heartbeat monitor exiting"
    ) &
    HEARTBEAT_MONITOR_PID=$!
    echo "[INFO] Started heartbeat monitor with PID: $HEARTBEAT_MONITOR_PID"
  fi

  # Wait for command to finish
  wait "$CMD_PID"
  CMD_EXIT=$?

  terminate_monitors

  # Calculate total runtime
  local END_TIME=$(date +%s)
  local DURATION=$((END_TIME - START_TIME))
  echo "[SUMMARY] Total runtime: $DURATION seconds ($((DURATION / 60)) minutes)"

  return $CMD_EXIT
}

cleanup() {
  # Only run cleanup once
  if [ "${CLEANUP_DONE:-}" = "true" ]; then
    return
  fi
  export CLEANUP_DONE=true

  # Capture the actual exit code that triggered the trap
  local actual_exit_code=$?

  # If CMD_EXIT wasn't set (meaning we failed before run_cmd), use the actual exit code
  if [ -z "${CMD_PID:-}" ]; then
    CMD_EXIT=$actual_exit_code
  fi

  # Read termination reason from file if it exists
  if [ -f "$TERMINATION_REASON_FILE" ]; then
    TERMINATION_REASON=$(cat "$TERMINATION_REASON_FILE")
    echo "[INFO] Termination reason from monitor: $TERMINATION_REASON"
  fi

  # Master-only: Handle notifications and status updates
  if [[ "$IS_MASTER" == "true" ]]; then
    # Check termination reason and set appropriate status
    if [[ "${TERMINATION_REASON}" == "heartbeat_timeout" ]]; then
      echo "[ERROR] Job terminated due to heartbeat timeout"
      export GITHUB_STATUS_STATE="failure"
      export GITHUB_STATUS_DESCRIPTION="Job failed - no heartbeat for ${HEARTBEAT_TIMEOUT} seconds"
      maybe_send_discord_notification "❌" "SkyPilot Job Failed" "${GITHUB_STATUS_DESCRIPTION}"

    elif [[ "${TERMINATION_REASON}" == "max_runtime_reached" ]]; then
      echo "[INFO] Job terminated due to max runtime limit"
      export GITHUB_STATUS_STATE="success"
      export GITHUB_STATUS_DESCRIPTION="Job ran successfully for ${MAX_RUNTIME_HOURS:-unknown} hours"
      # maybe_send_discord_notification "✅" "SkyPilot Job Completed" "${GITHUB_STATUS_DESCRIPTION}"
      # Map to success
      CMD_EXIT=0

    elif [[ "${TERMINATION_REASON}" == "force_restart_test" ]]; then
      echo "[INFO] Job restarting for test purposes"
      export GITHUB_STATUS_STATE="pending"
      export GITHUB_STATUS_DESCRIPTION="Forced a restart test in run #${RESTART_COUNT}"
      # maybe_send_discord_notification "🔄" "SkyPilot Job Restarting (Test)" "${GITHUB_STATUS_DESCRIPTION}"
      # Set exit code to trigger restart
      CMD_EXIT=1
      FINAL_EXIT_CODE=1

    elif [[ $CMD_EXIT -eq $EXIT_SUCCESS ]]; then
      echo "[SUCCESS] Job completed successfully"
      export GITHUB_STATUS_STATE="success"
      export GITHUB_STATUS_DESCRIPTION="Job completed successfully"
      export TERMINATION_REASON="completed"
      # maybe_send_discord_notification "✅" "SkyPilot Job Completed Successfully" "${GITHUB_STATUS_DESCRIPTION}"

    elif [[ $CMD_EXIT -eq $EXIT_NCCL_TEST_FAILURE ]]; then
      echo "[ERROR] Job failed during NCCL tests"
      export GITHUB_STATUS_STATE="error"  # Changed from "failure" - this is infrastructure issue
      export GITHUB_STATUS_DESCRIPTION="NCCL tests failed - GPU communication issue"
      export TERMINATION_REASON="nccl_test_failure"
      maybe_send_discord_notification "⚠️" "SkyPilot Job NCCL Config Error" "${GITHUB_STATUS_DESCRIPTION}"

    else
      echo "[ERROR] Job failed with exit code $CMD_EXIT"
      export GITHUB_STATUS_STATE="failure"
      export GITHUB_STATUS_DESCRIPTION="Job failed with exit code $CMD_EXIT"
      export TERMINATION_REASON="exit_code_${CMD_EXIT}"
      maybe_send_discord_notification "❌" "SkyPilot Job Failed" "${GITHUB_STATUS_DESCRIPTION}"
    fi

    # Final GitHub status update
    export CMD_EXIT
    maybe_set_github_status
  fi

  # Final summary (all nodes)
  echo "[SUMMARY] ===== Job Summary ====="
  echo "[SUMMARY] Node Rank: ${RANK}"
  echo "[SUMMARY] Metta Run ID: ${METTA_RUN_ID}"
  echo "[SUMMARY] Skypilot Task ID: ${SKYPILOT_TASK_ID}"
  echo "[SUMMARY] Exit code: ${CMD_EXIT}"
  echo "[SUMMARY] Termination reason: ${TERMINATION_REASON:-unknown}"
  echo "[SUMMARY] ======================"

  echo "[RUN] Job complete with exit code: $CMD_EXIT (reason: ${TERMINATION_REASON:-unknown})"

  # Set the final exit code for the script
  if [[ "${TERMINATION_REASON}" == "max_runtime_reached" ]] ||
     [[ "${TERMINATION_REASON}" == "completed" ]] ||
     [[ "${TERMINATION_REASON}" == "heartbeat_timeout" ]]; then
    echo "[INFO] Will exit with code 0 to prevent SkyPilot restart"
    FINAL_EXIT_CODE=0
  else
    echo "[INFO] Will exit with actual exit code: $CMD_EXIT"
    FINAL_EXIT_CODE=$CMD_EXIT
  fi

  # Worker nodes: brief delay to let master finish cleanup
  if [[ "$IS_MASTER" != "true" ]]; then
    echo "[INFO] Worker node waiting briefly for master cleanup..."
    sleep 3
  fi
}

# Export variables needed by cleanup
export TIMEOUT_MONITOR_PID=""
export HEARTBEAT_MONITOR_PID=""
export CMD_EXIT=1  # Default exit code
export FINAL_EXIT_CODE=1 # Default to failure
export -f terminate_process
export -f terminate_monitors

# Set up cleanup trap
trap cleanup EXIT

# All nodes: Run GPU diagnostics and NCCL tests (first start only)
if [ "${RESTART_COUNT:-0}" -eq 0 ]; then
  echo "[RUN] Running GPU diagnostics and NCCL tests (node ${RANK})..."
  uv run python ./devops/skypilot/test_nccl.py
else
  echo "[SKIP] Skipping NCCL test on restart (RESTART_COUNT=${RESTART_COUNT})"
fi

# Run the command
run_cmd
CMD_EXIT=$?

if [[ "$IS_MASTER" == "true" ]] && [ -z "${TERMINATION_REASON:-}" ] && [ -n "${CLUSTER_STOP_FILE:-}" ]; then
  echo "completed" > "$CLUSTER_STOP_FILE"
fi

# Exit with the appropriate code (cleanup will run automatically)
exit ${FINAL_EXIT_CODE:-$CMD_EXIT}
