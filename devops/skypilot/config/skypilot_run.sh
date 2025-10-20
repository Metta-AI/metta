#!/usr/bin/env bash

set -uo pipefail

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
  local termination_reason=$(cat "$TERMINATION_REASON_FILE" 2> /dev/null \
    || cat "$CLUSTER_STOP_FILE" 2> /dev/null \
    || echo "cluster_stop")
  echo "$termination_reason" > "$TERMINATION_REASON_FILE"

  # Kill the entire process tree
  if [ -n "${CMD_PGID:-}" ]; then
    echo "[SHUTDOWN] Initiating graceful shutdown of training process tree (PGID: ${CMD_PGID})"

    # Only master coordinates multi-node shutdown
    if [[ "$IS_MASTER" == "true" ]]; then
      echo "$termination_reason" > "$CLUSTER_STOP_FILE"
      echo "[SHUTDOWN] Master node signaled all nodes to begin shutdown"

      # Give workers time to detect the signal and start their own shutdown
      echo "[SHUTDOWN] Waiting for worker nodes to begin shutdown..."
      sleep 20

    else
      # Worker waits for cluster-wide shutdown signal
      echo "[SHUTDOWN] Worker node checking for cluster-wide shutdown signal..."
      count=0
      max_wait=20

      while [ ! -f "$CLUSTER_STOP_FILE" ] && [ $count -lt $max_wait ]; do
        sleep 1
        ((count++))
        if [ $((count % 5)) -eq 0 ]; then
          echo "[SHUTDOWN] Worker waiting for cluster signal... ${count}/${max_wait}s"
        fi
      done

      if [ -f "$CLUSTER_STOP_FILE" ]; then
        echo "[SHUTDOWN] Worker node detected cluster-wide shutdown signal"
      else
        echo "[SHUTDOWN] Worker node timeout waiting for cluster signal, proceeding with shutdown"
      fi
    fi

    kill -TERM -"${CMD_PGID}" 2> /dev/null || true
    sleep 20
    kill -KILL -"${CMD_PGID}" 2> /dev/null || true
  fi

  # shutdown now calls the cleanup_handler
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

  export START_TIME=$(date +%s)

  # Build the command as an array
  local cmd=(./devops/run.sh "${METTA_MODULE_PATH:?missing METTA_MODULE_PATH}")

  # Add args if METTA_ARGS is not empty
  if [ -n "${METTA_ARGS:-}" ]; then
    cmd+=(${METTA_ARGS}) # split on spaces
  fi

  echo "[INFO] Running command: ${cmd[*]}"

  # Use process substitution so $! is the trainer (not tee)
  setsid "${cmd[@]}" &
  export CMD_PID=$!

  sleep 1

  export CMD_PGID=$(ps -o pgid= -p "$CMD_PID" 2> /dev/null | tr -d ' ')
  echo "[INFO] Started process with PID: $CMD_PID, PGID: $CMD_PGID"

  start_monitors

  wait "$CMD_PID"
  CMD_EXIT=$?

  if [[ ! -f "$TERMINATION_REASON_FILE" ]] || [[ ! -s "$TERMINATION_REASON_FILE" ]]; then
    if [[ "$IS_MASTER" == "true" ]]; then
      echo "job_completed" > "$TERMINATION_REASON_FILE"
      echo "job_completed" > "$CLUSTER_STOP_FILE"
      echo "[INFO] Master wrote shutdown signal to cluster stop file"
    fi
  fi

  local END_TIME=$(date +%s)
  local DURATION=$((END_TIME - START_TIME))
  echo "[SUMMARY] Total runtime: $DURATION seconds ($((DURATION / 60)) minutes)"
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
shutdown
