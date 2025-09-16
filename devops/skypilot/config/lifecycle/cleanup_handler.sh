#!/usr/bin/env bash

set -euo pipefail

# Source the log helpers
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../monitors/log_helpers.sh"

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
    log_info "Termination reason: $TERMINATION_REASON"

    # Print debug report before cleanup
    print_job_debug_report
    sleep 0.1

    # Master-only: Handle notifications and status updates
    if [[ "$IS_MASTER" == "true" ]]; then
        print_final_summary
        # Force flush stdout to ensure summary is written
        exec 1>&1
        handle_master_cleanup
    fi

    # Set the final exit code for the script
    determine_final_exit_code

    # Final debug check
    log_debug "Final cleanup state check:"
    log_debug "Any remaining jobs:"
    jobs -l 2>&1 | sed 's/^/  /' || echo "  No jobs"

    sleep 1

    # Override the process exit code from within the EXIT trap.
    # Note: calling `exit` inside an EXIT trap does not recurse the trap.
    log_info "final exit code: ${FINAL_EXIT_CODE:-${CMD_EXIT:-1}}"
    exit "${FINAL_EXIT_CODE:-${CMD_EXIT:-1}}"
}

handle_master_cleanup() {
  case "${TERMINATION_REASON}" in

    "heartbeat_timeout")
      log_error "Job terminated due to heartbeat timeout"
      export GITHUB_STATUS_STATE="failure"
      export GITHUB_STATUS_DESCRIPTION="Job failed - no heartbeat for ${HEARTBEAT_TIMEOUT} seconds"
      bash ./devops/skypilot/config/observability/send_discord_notification.sh \
        "❌" "SkyPilot Job Failed" "${GITHUB_STATUS_DESCRIPTION}"
      ;;

    "max_runtime_reached")
      log_info "Job terminated due to max runtime limit"
      export GITHUB_STATUS_STATE="success"
      export GITHUB_STATUS_DESCRIPTION="Job ran successfully for ${MAX_RUNTIME_HOURS:-unknown} hours"
      CMD_EXIT=0
      ;;

    "force_restart_test")
      log_info "Job restarting for test purposes"
      export GITHUB_STATUS_STATE="pending"
      export GITHUB_STATUS_DESCRIPTION="Forced a restart test in run #${RESTART_COUNT}"
      CMD_EXIT=1
      FINAL_EXIT_CODE=1
      ;;

    "job_completed")
      log_info "SUCCESS: Job completed successfully"
      export GITHUB_STATUS_STATE="success"
      export GITHUB_STATUS_DESCRIPTION="Job completed successfully"
      CMD_EXIT=0
      ;;

    "")
      # Default fallback if no termination reason was written
      if [[ $CMD_EXIT -eq $EXIT_SUCCESS ]]; then
        log_info "SUCCESS: Job completed successfully (fallback)"
        export TERMINATION_REASON="job_completed"
        export GITHUB_STATUS_STATE="success"
        export GITHUB_STATUS_DESCRIPTION="Job completed successfully"
      else
        log_error "Job failed with exit code $CMD_EXIT (no termination reason)"
        export TERMINATION_REASON="exit_code_${CMD_EXIT}"
        export GITHUB_STATUS_STATE="failure"
        export GITHUB_STATUS_DESCRIPTION="Job failed with exit code $CMD_EXIT"
      fi
      ;;

    *)
      if [[ $CMD_EXIT -eq $EXIT_NCCL_TEST_FAILURE ]]; then
        log_error "Job failed during NCCL tests"
        export GITHUB_STATUS_STATE="error"
        export GITHUB_STATUS_DESCRIPTION="NCCL tests failed - GPU communication issue"
        export TERMINATION_REASON="nccl_test_failure"
        bash ./devops/skypilot/config/observability/send_discord_notification.sh \
          "⚠️" "SkyPilot Job NCCL Config Error" "${GITHUB_STATUS_DESCRIPTION}"
      else
        log_error "Job failed with exit code $CMD_EXIT (reason: $TERMINATION_REASON)"
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
  log_info "===== Job Summary ====="
  log_info "Metta Run ID: ${METTA_RUN_ID}"
  log_info "Skypilot Task ID: ${SKYPILOT_TASK_ID}"
  log_info "Restart Count: ${RESTART_COUNT}"
  log_info "Exit code: ${CMD_EXIT}"
  log_info "Termination reason: ${TERMINATION_REASON:-unknown}"
  log_info "======================"

  log_info "RUN: Job complete with exit code: $CMD_EXIT (reason: ${TERMINATION_REASON:-unknown})"
}

determine_final_exit_code() {
  if [[ "${TERMINATION_REASON}" == "max_runtime_reached" ]] \
    || [[ "${TERMINATION_REASON}" == "job_completed" ]] \
    || [[ "${TERMINATION_REASON}" == "job_failed" ]] \
    || [[ "${TERMINATION_REASON}" == "heartbeat_timeout" ]]; then
    log_info "Will exit with code 0 to prevent SkyPilot restart"
    FINAL_EXIT_CODE=0
  else
    log_info "Will exit with code: $CMD_EXIT"
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
