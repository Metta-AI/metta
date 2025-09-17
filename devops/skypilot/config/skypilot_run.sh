#!/usr/bin/env bash

set -euo pipefail

cd /workspace/metta

# Drop any preloaded venv; activate your own
if [ -n "${VIRTUAL_ENV:-}" ]; then
  deactivate 2> /dev/null || true
fi
. .venv/bin/activate

export WRAPPER_PID=$BASHPID

# Determine node role using SkyPilot environment variables
export RANK=${SKYPILOT_NODE_RANK:-0}
export IS_MASTER=$([[ "$RANK" == "0" ]] && echo "true" || echo "false")
TOTAL_NODES=${SKYPILOT_NUM_NODES:-1}

DEBUG=${DEBUG:-0}

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
echo "  - METTA_MODULE_PATH: ${METTA_MODULE_PATH:-'NOT SET'}"
echo "  - METTA_ARGS: ${METTA_ARGS:-'NOT SET'}"
[ "$DEBUG" = "1" ] && echo "  - DEBUG: ENABLED"

# Master-only: Collect SkyPilot latency
if [[ "$IS_MASTER" == "true" ]]; then
  if [ -f devops/skypilot/utils/job_latency.py ]; then
    echo "[RUN] Collecting skypilot latency..."
    uv run python devops/skypilot/utils/job_latency.py || true
  else
    echo "[RUN] Latency script is missing!"
  fi
fi

METTA_ENV_FILE="$(uv run ./common/src/metta/common/util/constants.py METTA_ENV_FILE)"

# Master-only: Collect instance cost
if [[ "$IS_MASTER" == "true" ]]; then
  if [ -f devops/skypilot/utils/cost_monitor.py ]; then
    echo "[RUN] Collecting instance cost..."
    if uv run python devops/skypilot/utils/cost_monitor.py; then
      source "$METTA_ENV_FILE"
      echo "[RUN] METTA_HOURLY_COST set to: $METTA_HOURLY_COST"
    else
      echo "[RUN] Cost monitor script failed to run."
    fi
  else
    echo "[RUN] Cost monitor script is missing!"
  fi
fi

# Setup environment (all nodes)
bash ./devops/skypilot/config/lifecycle/configure_environment.sh
source "$METTA_ENV_FILE"

export EXIT_SUCCESS=0
export EXIT_FAILURE=1
export EXIT_NCCL_TEST_FAILURE=42

# Compute derived runtime values
max_seconds=-1 # no max runtime
remaining_at_start=-1
force_restart_seconds=-1
if [[ -n "${MAX_RUNTIME_HOURS:-}" ]]; then
  max_seconds=$(awk "BEGIN {print int(${MAX_RUNTIME_HOURS} * 3600)}")
  accumulated_runtime=${ACCUMULATED_RUNTIME:-0}
  remaining_at_start=$((max_seconds - accumulated_runtime))
  if [[ ${RESTART_COUNT:-0} -eq 0 ]]; then
    force_restart_seconds=$(awk "BEGIN {print int(${remaining_at_start} * 0.3)}")
  fi
fi

# Print run configuration
echo "[CONFIG] Run Configuration:"
echo "  - WRAPPER_PID: $WRAPPER_PID"
echo "  - NODE_RANK: ${RANK}"
echo "  - IS_MASTER: ${IS_MASTER}"
echo "  - TOTAL_NODES: ${TOTAL_NODES}"
echo "  - METTA_RUN_ID: ${METTA_RUN_ID:-}"
echo "  - SKYPILOT_TASK_ID: ${SKYPILOT_TASK_ID:-}"
echo "  - HEARTBEAT_TIMEOUT: ${HEARTBEAT_TIMEOUT:-'NOT SET'}"
echo "  - MAX_RUNTIME_HOURS: ${MAX_RUNTIME_HOURS:-'NOT SET'}"
echo "  - ACCUMULATED_RUNTIME: ${ACCUMULATED_RUNTIME:-'NOT SET'}"
[[ ${remaining_at_start} -gt 0 ]] && echo "     ↳ remaining runtime seconds: ${remaining_at_start}"
echo "  - RESTART_COUNT: ${RESTART_COUNT}"
echo "  - TEST_JOB_RESTART: ${TEST_JOB_RESTART:-false}" # used in timeout_monitor
[[ ${force_restart_seconds} -gt 0 ]] && echo "     ↳ job restart test delay: ${force_restart_seconds}"
echo "  - TEST_NCCL: ${TEST_NCCL:-false}"
[[ "${TEST_NCCL:-false}" == "true" ]] && [[ "${RESTART_COUNT:-0}" -eq 0 ]] && echo " ↳ will run"
[[ "${TEST_NCCL:-false}" == "true" ]] && [[ "${RESTART_COUNT:-0}" -gt 0 ]] && echo " ↳ skipping on restart #${RESTART_COUNT}"
echo "  - JOB_METADATA_DIR: $JOB_METADATA_DIR"
echo "     ↳ TERMINATION_REASON_FILE: $TERMINATION_REASON_FILE"
echo "     ↳ CLUSTER_STOP_FILE: $CLUSTER_STOP_FILE"
echo "     ↳ HEARTBEAT_FILE: $HEARTBEAT_FILE"
echo "     ↳ ACCUMULATED_RUNTIME_FILE: $ACCUMULATED_RUNTIME_FILE"

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

    # Set initial GitHub status
    uv run devops/skypilot/config/observability/set_github_status.py "pending" "Queued on SkyPilot…"
  else
    export ENABLE_GITHUB_STATUS=false
    echo "[RUN] GitHub status reporting is disabled (missing required credentials)"
  fi
else
  export ENABLE_DISCORD=false
  export ENABLE_GITHUB_STATUS=false
fi

shutdown() {
  # Disable the trap to prevent re-entry
  trap '' INT TERM HUP

  echo "[SHUTDOWN] Caught INT/TERM/HUP; initiating graceful shutdown..."

  # Get termination reason
  local termination_reason=$(cat "$TERMINATION_REASON_FILE" 2>/dev/null ||
                            cat "$CLUSTER_STOP_FILE" 2>/dev/null ||
                            echo "cluster_stop")
  echo "$termination_reason" > "$TERMINATION_REASON_FILE"

  # Only proceed if we have a process to kill
  [ -z "${CMD_PID:-}" ] && exit 0

  echo "[SHUTDOWN] Initiating shutdown of process tree (PID: ${CMD_PID})"

  # Coordinate multi-node shutdown
  if [[ "$IS_MASTER" == "true" ]]; then
    echo "$termination_reason" > "$CLUSTER_STOP_FILE"
    echo "[SHUTDOWN] Master signaled all nodes to begin shutdown"
    sleep 20  # Give workers time to detect signal
  else
    # Worker waits for cluster-wide shutdown signal (max 20s)
    echo "[SHUTDOWN] Worker waiting for cluster-wide shutdown signal..."
    local count=0
    while [ ! -f "$CLUSTER_STOP_FILE" ] && [ $count -lt 20 ]; do
      sleep 1
      ((count++))
      [ $((count % 5)) -eq 0 ] && echo "[SHUTDOWN] Still waiting... ${count}/20s"
    done
    echo "[SHUTDOWN] $([ -f "$CLUSTER_STOP_FILE" ] && echo "Detected" || echo "Timeout on") cluster signal"
  fi

  # Helper function to wait for process death
  wait_for_exit() {
    local pid=$1 sig=$2 max_wait=$3
    local count=0

    while kill -0 "$pid" 2>/dev/null && [ $count -lt $max_wait ]; do
      sleep 1
      ((count++))
      [ $((count % 5)) -eq 0 ] && echo "[SHUTDOWN] Waiting for $sig shutdown... ${count}/${max_wait}s"
    done

    # Return 0 if process died, 1 if still alive
    ! kill -0 "$pid" 2>/dev/null
  }

  # Try graceful shutdown first
  kill -TERM -"${CMD_PID}" 2>/dev/null || true

  if ! wait_for_exit "$CMD_PID" "graceful" 30; then
    echo "[SHUTDOWN] Process didn't terminate gracefully, using SIGKILL"
    kill -KILL -"${CMD_PID}" 2>/dev/null || true
    wait_for_exit "$CMD_PID" "forced" 10 || echo "[SHUTDOWN] WARNING: Process may still be running"
  fi

  echo "[SHUTDOWN] Process $CMD_PID shutdown complete"
  exit 0
}

trap shutdown INT TERM HUP

start_monitors() {
  if [[ -n "${HEARTBEAT_TIMEOUT:-}" ]]; then
    bash ./devops/skypilot/config/monitors/heartbeat_monitor.sh &
    echo "[INFO] Started heartbeat monitor"
  fi
  if [[ "$IS_MASTER" == "true" ]] && [[ -n "${MAX_RUNTIME_HOURS:-}" ]]; then
    bash ./devops/skypilot/config/monitors/timeout_monitor.sh &
    echo "[INFO] Started timeout monitor"
  fi
  if [[ -n "${CLUSTER_STOP_FILE:-}" ]]; then
    bash ./devops/skypilot/config/monitors/cluster_stop_monitor.sh &
    echo "[INFO] Started cluster-stop monitor"
  fi
  if [[ "$IS_MASTER" == "true" ]] && [[ "${TEST_JOB_RESTART:-false}" == "true" ]]; then
    bash ./devops/skypilot/config/monitors/test_job_restart_monitor.sh &
    echo "[INFO] Started test job restart monitor"
  fi
}

run_cmd() {
  echo "[INFO] Starting process (node rank: $RANK)"
  local START_TIME=$(date +%s)

  # Enable job control
  set -m

  # Build and run command
  local cmd=(./devops/run.sh "${METTA_MODULE_PATH:?missing METTA_MODULE_PATH}")
  [ -n "${METTA_ARGS:-}" ] && cmd+=(${METTA_ARGS})

  echo "[INFO] Running command: ${cmd[*]}"
  "${cmd[@]}" &
  export CMD_PID=$!  # Only export needed for monitors

  echo "[INFO] Started process with PID: $CMD_PID"
  start_monitors

  # Wait for process to exit
  while kill -0 "$CMD_PID" 2>/dev/null; do
      sleep 1
  done

  # Get exit code from job status
  local JOB_INFO=$(jobs -l %1 2>&1 || echo "")
  local CMD_EXIT=1  # Default to failure

  if [[ "$JOB_INFO" =~ Exit\ ([0-9]+) ]]; then
      CMD_EXIT=${BASH_REMATCH[1]}
  elif [[ "$JOB_INFO" =~ Done ]]; then
      CMD_EXIT=0
  fi

  echo "[INFO] Process exited with code $CMD_EXIT"
  set +m

  # Handle completion - only write job_completed if actually successful
  if [[ ! -s "${TERMINATION_REASON_FILE:-}" ]] && [[ "$IS_MASTER" == "true" ]] && [[ $CMD_EXIT -eq 0 ]]; then
        echo "job_completed" > "$TERMINATION_REASON_FILE"
        echo "job_completed" > "$CLUSTER_STOP_FILE"
        echo "[INFO] Master wrote shutdown signal"
  fi

  local DURATION=$(($(date +%s) - START_TIME))
  echo "[SUMMARY] Total runtime: $DURATION seconds ($((DURATION / 60)) minutes)"

  return $CMD_EXIT
}

source ./devops/skypilot/config/lifecycle/cleanup_handler.sh
trap cleanup EXIT

# All nodes: Run GPU diagnostics and NCCL tests (first start only)
TEST_NCCL="${TEST_NCCL:-false}"
if [[ "$TEST_NCCL" == "false" ]]; then
  echo "[SKIP] Skipping NCCL test (TEST_NCCL=false)"
elif [ "${RESTART_COUNT:-0}" -ne 0 ]; then
  echo "[SKIP] Skipping NCCL test on restarted job (RESTART_COUNT=${RESTART_COUNT})"
else
  echo "[RUN] Running GPU diagnostics and NCCL tests (node ${RANK})..."

  # Run the test in a subshell to isolate it
  NCCL_TEST_EXIT_CODE=0
  (
    uv run python ./devops/skypilot/config/preflight/test_nccl.py
  ) || NCCL_TEST_EXIT_CODE=$?

  sleep 10 # wait for other nodes to complete tests

  if [ $NCCL_TEST_EXIT_CODE -ne 0 ]; then
    echo "[ERROR] NCCL tests failed with exit code: $NCCL_TEST_EXIT_CODE"
    echo "nccl_test_failure" > "$TERMINATION_REASON_FILE"
    kill -TERM "${WRAPPER_PID}" 2> /dev/null || true # initiate shutdown
  else
    echo "[SUCCESS] NCCL tests passed"
  fi
fi

run_cmd
