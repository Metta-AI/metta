#!/usr/bin/env bash

set -euo pipefail

# Required environment variables (should be exported by parent script)
: "${IS_MASTER:?Missing IS_MASTER}"
: "${RANK:?Missing RANK}"
: "${METTA_RUN_ID:?Missing METTA_RUN_ID}"
: "${SKYPILOT_TASK_ID:?Missing SKYPILOT_TASK_ID}"
: "${TERMINATION_REASON_FILE:?Missing TERMINATION_REASON_FILE}"

# Exit code constants
: "${EXIT_SUCCESS:?Missing EXIT_SUCCESS}"
: "${EXIT_FAILURE:?Missing EXIT_FAILURE}"
: "${EXIT_NCCL_TEST_FAILURE:?Missing EXIT_NCCL_TEST_FAILURE}"

cleanup() {
  # Capture the actual exit code that triggered the trap
  CMD_EXIT=${CMD_EXIT:-$?}

  # Read termination reason
  TERMINATION_REASON=$(cat "$TERMINATION_REASON_FILE" 2>/dev/null || echo "")
  echo "[INFO] Termination reason: $TERMINATION_REASON"

  # Master-only: Handle notifications and status updates
  if [[ "$IS_MASTER" == "true" ]]; then
    print_final_summary
    # Force flush stdout to ensure summary is written
    exec 1>&1

    handle_master_cleanup
  fi

  # Set the final exit code for the script
  determine_final_exit_code

  sleep 1

  print_job_debug_report

  # Override the process exit code from within the EXIT trap.
  # Note: calling `exit` inside an EXIT trap does not recurse the trap.
  exit "${FINAL_EXIT_CODE:-${CMD_EXIT:-1}}"
}

handle_master_cleanup() {
  case "${TERMINATION_REASON}" in

    "heartbeat_timeout")
      echo "[ERROR] Job terminated due to heartbeat timeout"
      export GITHUB_STATUS_STATE="failure"
      export GITHUB_STATUS_DESCRIPTION="Job failed - no heartbeat for ${HEARTBEAT_TIMEOUT} seconds"
      bash ./devops/skypilot/config/observability/send_discord_notification.sh \
        "❌" "SkyPilot Job Failed" "${GITHUB_STATUS_DESCRIPTION}"
      ;;

    "max_runtime_reached")
      echo "[INFO] Job terminated due to max runtime limit"
      export GITHUB_STATUS_STATE="success"
      export GITHUB_STATUS_DESCRIPTION="Job ran successfully for ${MAX_RUNTIME_HOURS:-unknown} hours"
      CMD_EXIT=0
      ;;

    "force_restart_test")
      echo "[INFO] Job restarting for test purposes"
      export GITHUB_STATUS_STATE="pending"
      export GITHUB_STATUS_DESCRIPTION="Forced a restart test in run #${RESTART_COUNT}"
      CMD_EXIT=1
      FINAL_EXIT_CODE=1
      ;;

    "job_completed")
      echo "[SUCCESS] Job completed successfully"
      export GITHUB_STATUS_STATE="success"
      export GITHUB_STATUS_DESCRIPTION="Job completed successfully"
      CMD_EXIT=0
      ;;

    "")
      # Default fallback if no termination reason was written
      if [[ $CMD_EXIT -eq $EXIT_SUCCESS ]]; then
        echo "[SUCCESS] Job completed successfully (fallback)"
        export TERMINATION_REASON="completed"
        export GITHUB_STATUS_STATE="success"
        export GITHUB_STATUS_DESCRIPTION="Job completed successfully"
      else
        echo "[ERROR] Job failed with exit code $CMD_EXIT (no termination reason)"
        export TERMINATION_REASON="exit_code_${CMD_EXIT}"
        export GITHUB_STATUS_STATE="failure"
        export GITHUB_STATUS_DESCRIPTION="Job failed with exit code $CMD_EXIT"
      fi
      ;;

    *)
      if [[ $CMD_EXIT -eq $EXIT_NCCL_TEST_FAILURE ]]; then
        echo "[ERROR] Job failed during NCCL tests"
        export GITHUB_STATUS_STATE="error"
        export GITHUB_STATUS_DESCRIPTION="NCCL tests failed - GPU communication issue"
        export TERMINATION_REASON="nccl_test_failure"
        bash ./devops/skypilot/config/observability/send_discord_notification.sh \
          "⚠️" "SkyPilot Job NCCL Config Error" "${GITHUB_STATUS_DESCRIPTION}"
      else
        echo "[ERROR] Job failed with exit code $CMD_EXIT (reason: $TERMINATION_REASON)"
        export GITHUB_STATUS_STATE="failure"
        export GITHUB_STATUS_DESCRIPTION="Job failed with exit code $CMD_EXIT"
        export TERMINATION_REASON="exit_code_${CMD_EXIT}"
        bash ./devops/skypilot/config/observability/send_discord_notification.sh \
          "❌" "SkyPilot Job Failed" "${GITHUB_STATUS_DESCRIPTION}"
      fi
      ;;
  esac

  export CMD_EXIT
  uv run devops/skypilot/config/observability/set_github_status.py "$GITHUB_STATUS_STATE" "$GITHUB_STATUS_DESCRIPTION"
}


print_final_summary() {
  echo "[SUMMARY] ===== Job Summary ====="
  echo "[SUMMARY] Metta Run ID: ${METTA_RUN_ID}"
  echo "[SUMMARY] Skypilot Task ID: ${SKYPILOT_TASK_ID}"
  echo "[SUMMARY] Restart Count: ${RESTART_COUNT}"
  echo "[SUMMARY] Exit code: ${CMD_EXIT}"
  echo "[SUMMARY] Termination reason: ${TERMINATION_REASON:-unknown}"
  echo "[SUMMARY] ======================"

  echo "[RUN] Job complete with exit code: $CMD_EXIT (reason: ${TERMINATION_REASON:-unknown})"
}

determine_final_exit_code() {
  echo "[DEBUG] TERMINATION_REASON='${TERMINATION_REASON}'"

  if [[ "${TERMINATION_REASON}" == "force_restart_test" ]]; then
    echo "[INFO] Will exit with code 1 to trigger SkyPilot restart"
    FINAL_EXIT_CODE=1
  else
    echo "[INFO] Will exit with code 0 to prevent SkyPilot restart"
    FINAL_EXIT_CODE=0
  fi

  echo "[DEBUG] FINAL_EXIT_CODE=${FINAL_EXIT_CODE}"
}

print_job_debug_report() {
    local report=""

    report+="[DEBUG] ========== JOB DEBUG REPORT ==========\n"
    report+="[DEBUG] Timestamp: $(date '+%Y-%m-%d %H:%M:%S')\n"
    report+="[DEBUG] Script PID: $$\n"
    report+="[DEBUG] CMD_PID: ${CMD_PID:-'not set'}\n"

    # Check current job status
    report+="[DEBUG] Current job table:\n"
    report+="$(jobs -l 2>&1 || echo '  No jobs found')\n"

    # Check all background jobs
    report+="[DEBUG] Background job PIDs:\n"
    report+="$(jobs -p 2>&1 || echo '  No background jobs')\n"

    # Check process group (python -m mode has PGID = CMD_PID)
    report+="[DEBUG] Process group ${CMD_PID} members:\n"
    report+="$(ps -eo pid,ppid,pgid,state,etime,cmd | grep "^\s*[0-9]\+\s\+[0-9]\+\s\+${CMD_PID}" 2>&1 || echo '  No processes found in group')\n"

    # Check for any child processes of this script
    report+="[DEBUG] Child processes of script (PID $$):\n"
    report+="$(ps --ppid $$ -o pid,ppid,pgid,state,etime,cmd 2>&1 || echo '  No child processes found')\n"

    # Check for zombie processes
    report+="[DEBUG] Zombie processes:\n"
    report+="$(ps aux | grep ' <defunct>' | grep -v grep || echo '  No zombie processes found')\n"

    # Check for processes matching the module name
    if [[ -n "${METTA_MODULE_PATH:-}" ]]; then
        local module_name=$(basename "${METTA_MODULE_PATH}")
        report+="[DEBUG] Processes matching module '$module_name':\n"
        report+="$(ps aux | grep "$module_name" | grep -v grep || echo '  No matching processes found')\n"
    fi

    # Check monitor processes if they exist
    report+="[DEBUG] Monitor processes:\n"
    report+="$(ps aux | grep -E '(monitor_gpu|monitor_memory|cluster_stop_monitor)' | grep -v grep || echo '  No monitor processes found')\n"

    report+="[DEBUG] ========== END DEBUG REPORT ==========\n"

    # Output everything at once
    echo -ne "$report"
}
