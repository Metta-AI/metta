#!/usr/bin/env bash
# cleanup_handler.sh - Modular cleanup functionality for sk_train_run

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
  TERMINATION_REASON=$(cat "$TERMINATION_REASON_FILE")
  echo "[INFO] Termination reason: $TERMINATION_REASON"

  # Master-only: Handle notifications and status updates
  if [[ "$IS_MASTER" == "true" ]]; then
    handle_master_cleanup
    print_final_summary
  fi

  # Set the final exit code for the script
  determine_final_exit_code

  sleep 1

  # Override the process exit code from within the EXIT trap.
  # Note: calling `exit` inside an EXIT trap does not recurse the trap.
  exit "${FINAL_EXIT_CODE:-${CMD_EXIT:-1}}"
}

handle_master_cleanup() {
  # Check termination reason and set appropriate status
  if [[ "${TERMINATION_REASON}" == "heartbeat_timeout" ]]; then
    echo "[ERROR] Job terminated due to heartbeat timeout"
    export GITHUB_STATUS_STATE="failure"
    export GITHUB_STATUS_DESCRIPTION="Job failed - no heartbeat for ${HEARTBEAT_TIMEOUT} seconds"
    bash ./devops/skypilot/config/observability/send_discord_notification.sh \
      "❌" "SkyPilot Job Failed" "${GITHUB_STATUS_DESCRIPTION}"

  elif [[ "${TERMINATION_REASON}" == "max_runtime_reached" ]]; then
    echo "[INFO] Job terminated due to max runtime limit"
    export GITHUB_STATUS_STATE="success"
    export GITHUB_STATUS_DESCRIPTION="Job ran successfully for ${MAX_RUNTIME_HOURS:-unknown} hours"
    # bash ./devops/skypilot/config/observability/send_discord_notification.sh \
    #   "✅" "SkyPilot Job Completed" "${GITHUB_STATUS_DESCRIPTION}"
    # Map to success
    CMD_EXIT=0

  elif [[ "${TERMINATION_REASON}" == "force_restart_test" ]]; then
    echo "[INFO] Job restarting for test purposes"
    export GITHUB_STATUS_STATE="pending"
    export GITHUB_STATUS_DESCRIPTION="Forced a restart test in run #${RESTART_COUNT}"
    # bash ./devops/skypilot/config/observability/send_discord_notification.sh \
    #   "🔄" "SkyPilot Job Restarting (Test)" "${GITHUB_STATUS_DESCRIPTION}"
    # Set exit code to trigger restart
    CMD_EXIT=1
    FINAL_EXIT_CODE=1

  elif [[ -z "${TERMINATION_REASON}" ]]; then
    if [[ $CMD_EXIT -eq $EXIT_SUCCESS ]]; then
      echo "[SUCCESS] Job completed successfully"
      export TERMINATION_REASON="completed"
      export GITHUB_STATUS_STATE="success"
      export GITHUB_STATUS_DESCRIPTION="Job completed successfully"
      # bash ./devops/skypilot/config/observability/send_discord_notification.sh \
      #   "✅" "SkyPilot Job Completed Successfully" "${GITHUB_STATUS_DESCRIPTION}"
    else
      echo "[ERROR] Job failed with exit code $CMD_EXIT"
      export TERMINATION_REASON="exit_code_${CMD_EXIT}"
      export GITHUB_STATUS_STATE="failure"
      export GITHUB_STATUS_DESCRIPTION="Job failed with exit code $CMD_EXIT"
    fi

  elif [[ $CMD_EXIT -eq $EXIT_NCCL_TEST_FAILURE ]]; then
    echo "[ERROR] Job failed during NCCL tests"
    export GITHUB_STATUS_STATE="error" # Changed from "failure" - this is infrastructure issue
    export GITHUB_STATUS_DESCRIPTION="NCCL tests failed - GPU communication issue"
    export TERMINATION_REASON="nccl_test_failure"
    bash ./devops/skypilot/config/observability/send_discord_notification.sh \
      "⚠️" "SkyPilot Job NCCL Config Error" "${GITHUB_STATUS_DESCRIPTION}"

  else
    echo "[ERROR] Job failed with exit code $CMD_EXIT"
    export GITHUB_STATUS_STATE="failure"
    export GITHUB_STATUS_DESCRIPTION="Job failed with exit code $CMD_EXIT"
    export TERMINATION_REASON="exit_code_${CMD_EXIT}"
    bash ./devops/skypilot/config/observability/send_discord_notification.sh \
      "❌" "SkyPilot Job Failed" "${GITHUB_STATUS_DESCRIPTION}"
  fi

  # Final GitHub status update
  export CMD_EXIT

  uv run devops/skypilot/config/observability/set_github_status.py "$GITHUB_STATUS_STATE" "$GITHUB_STATUS_DESCRIPTION"
}

print_final_summary() {
  echo "[SUMMARY] ===== Job Summary ====="
  echo "[SUMMARY] Metta Run ID: ${METTA_RUN_ID}"
  echo "[SUMMARY] Skypilot Task ID: ${SKYPILOT_TASK_ID}"
  echo "[SUMMARY] Exit code: ${CMD_EXIT}"
  echo "[SUMMARY] Termination reason: ${TERMINATION_REASON:-unknown}"
  echo "[SUMMARY] ======================"

  echo "[RUN] Job complete with exit code: $CMD_EXIT (reason: ${TERMINATION_REASON:-unknown})"
}

determine_final_exit_code() {
  if [[ "${TERMINATION_REASON}" == "max_runtime_reached" ]] \
    || [[ "${TERMINATION_REASON}" == "completed" ]] \
    || [[ "${TERMINATION_REASON}" == "heartbeat_timeout" ]]; then
    echo "[INFO] Will exit with code 0 to prevent SkyPilot restart"
    FINAL_EXIT_CODE=0
  else
    echo "[INFO] Will exit with code: $CMD_EXIT"
    FINAL_EXIT_CODE=$CMD_EXIT
  fi
}

# Export the cleanup function if sourced
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  export -f cleanup
  export -f handle_master_cleanup
  export -f print_final_summary
  export -f determine_final_exit_code
fi
