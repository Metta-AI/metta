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
    METTA_HOURLY_COST="$(uv run python common/src/metta/common/util/cost_monitor.py 2> /dev/null | tail -1 || true)"
    echo "[RUN] METTA_HOURLY_COST set to $METTA_HOURLY_COST in $METTA_ENV_FILE by python."
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
[[ "${RESTART_COUNT:-0}" -eq 0 ]] && echo "     ↳ will be run"
[[ "${RESTART_COUNT:-0}" -gt 0 ]] && echo "     ↳ skipping on restart #${RESTART_COUNT}"
echo "  - JOB_METADATA_DIR: $JOB_METADATA_DIR"
echo "     ↳ TERMINATION_REASON_FILE: $TERMINATION_REASON_FILE"
echo "     ↳ CLUSTER_STOP_FILE: $CLUSTER_STOP_FILE"
echo "     ↳ HEARTBEAT_FILE: $HEARTBEAT_FILE"
echo "     ↳ ACCUMULATED_RUNTIME_FILE: $ACCUMULATED_RUNTIME_FILE"
echo "  - METTA_CMD: ${METTA_CMD:-'NOT SET'}"
echo "  - METTA_CMD_ARGS: ${METTA_CMD_ARGS:-'NOT SET'}"

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

  local termination_reason="$(cat "$TERMINATION_REASON_FILE" || true)"

  # Kill the entire process tree gracefully
  if [ -n "${CMD_PGID:-}" ] && [ -n "${CMD_PID:-}" ]; then
    echo "[SHUTDOWN] Initiating graceful shutdown of training process tree (PGID: ${CMD_PGID})"

    # First, signal all worker nodes to start shutdown
    if [[ "$IS_MASTER" == "true" ]] && [[ "$TOTAL_NODES" -gt 1 ]]; then
      echo "$termination_reason" > "$CLUSTER_STOP_FILE"
      echo "[SHUTDOWN] Signaled all nodes to begin shutdown"
      sleep 20 # Give workers time to shut down
    fi

    # Send SIGTERM to the process group
    kill -TERM -"${CMD_PGID}" 2> /dev/null || true

    # Wait longer for graceful shutdown (especially for distributed training)
    count=0
    max_wait=30
    while kill -0 "$CMD_PID" 2> /dev/null && [ $count -lt $max_wait ]; do
      sleep 1
      ((count++))
      if [ $((count % 5)) -eq 0 ]; then
        echo "[SHUTDOWN] Waiting for graceful shutdown... ${count}/${max_wait}s"
      fi
    done

    # If STILL alive, use SIGKILL
    if kill -0 "$CMD_PID" 2> /dev/null; then
      echo "[SHUTDOWN] Process didn't terminate gracefully after ${max_wait}s, using SIGKILL"
      kill -KILL -"${CMD_PGID}" 2> /dev/null || true
    fi

    # Wait for the process to actually exit
    if kill -0 "$CMD_PID" 2> /dev/null; then
      echo "[SHUTDOWN] Waiting for process $CMD_PID to exit..."
      wait "$CMD_PID" 2> /dev/null || true
    else
      echo "[SHUTDOWN] Process $CMD_PID already exited"
    fi
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
  echo "[INFO] Starting process $METTA_CMD (node rank: $RANK)"

  export START_TIME=$(date +%s)

  # Start training in its own process group; tee output for postmortem
  cmd=(./devops/"${METTA_CMD:?missing METTA_CMD}".sh "run=${METTA_RUN_ID:?missing METTA_RUN_ID}")
  if [ -n "${METTA_CMD_ARGS:-}" ]; then
    extra_args=(${METTA_CMD_ARGS})
    cmd+=("${extra_args[@]}")
  fi
  # Use process substitution so $! is the trainer (not tee)
  setsid "${cmd[@]}" &
  export CMD_PID=$!

  sleep 1

  export CMD_PGID=$(ps -o pgid= -p "$CMD_PID" 2> /dev/null | tr -d ' ')
  echo "[INFO] Started $METTA_CMD process with PID: $CMD_PID, PGID: $CMD_PGID"

  start_monitors

  wait "$CMD_PID"
  CMD_EXIT=$?

  local END_TIME=$(date +%s)
  local DURATION=$((END_TIME - START_TIME))
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

  # Run the test and capture the exit code
  set +e
  uv run python ./devops/skypilot/config/preflight/test_nccl.py
  NCCL_TEST_EXIT_CODE=$?
  set -e

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
