#!/bin/bash
# Launch LP performance optimization tests on cloud infrastructure
# Uses skypilot/launch.py to run all test configurations

set -e

echo "=========================================="
echo "LP Performance Matrix - Cloud Launch"
echo "=========================================="
echo ""

# Configuration
GPUS=${GPUS:-1}
SPOT=${SPOT:-"--spot"}
MAX_RUNTIME=${MAX_RUNTIME:-24} # hours
GIT_REF=${GIT_REF:-$(git rev-parse --abbrev-ref HEAD)}

echo "Launch Configuration:"
echo "  GPUs: $GPUS"
echo "  Spot instances: ${SPOT:---no-spot}"
echo "  Max runtime: ${MAX_RUNTIME}h"
echo "  Git ref: $GIT_REF"
echo ""

# Test configurations
declare -a CONFIGS=(
  "baseline:1:False"
  "batch_10:10:True"
  "batch_100:100:True"
  "batch_1000:1000:True"
  "cache_only:1:True"
  "small_pool_100:100:True:100"
)

# Function to launch a single test
launch_test() {
  local config_line=$1
  IFS=':' read -r name batch_size cache pool_size <<< "$config_line"

  echo ""
  echo "===================================="
  echo "Launching: msb_perfdiagnosis_${name}"
  echo "===================================="
  echo "  Batch size: $batch_size"
  echo "  Task list cache: $cache"
  echo "  Pool size: ${pool_size:-1000}"
  echo ""

  # Build the command arguments
  local args=(
    "cogs_v_clips_perf_test.train_with_perf_config"
    "--gpus" "$GPUS"
    "--max-runtime-hours" "$MAX_RUNTIME"
    "--git-ref" "$GIT_REF"
  )

  # Add spot flag if enabled
  if [ "$SPOT" == "--spot" ]; then
    args+=("--spot")
  fi

  # Add separator before tool args
  args+=("--")

  # Tool arguments (passed to train_with_perf_config function)
  args+=(
    "run=msb_perfdiagnosis_${name}"
    "use_lp=True"
    "perf_invalidation_batch_size=${batch_size}"
    "perf_cache_task_list=${cache}"
    "perf_log_metrics=True"
  )

  # Add pool size if specified
  if [ -n "$pool_size" ]; then
    args+=("num_active_tasks=${pool_size}")
  fi

  # Launch the job
  echo "Command: devops/skypilot/launch.py ${args[*]}"
  echo ""

  if [ "${DRY_RUN}" == "true" ]; then
    echo "[DRY RUN] Would launch: ${args[*]}"
  else
    uv run python devops/skypilot/launch.py "${args[@]}" || {
      echo "ERROR: Failed to launch msb_perfdiagnosis_${name}"
      echo "Continuing with remaining tests..."
    }
  fi

  echo ""
  sleep 2 # Brief pause between launches
}

# Parse command line arguments
MODE=${1:-all}

case "$MODE" in
  "all")
    echo "Launching all performance tests"
    echo "Total jobs: ${#CONFIGS[@]}"
    echo ""

    for config in "${CONFIGS[@]}"; do
      launch_test "$config"
    done
    ;;

  "quick")
    echo "Launching quick comparison (baseline, batch_10, batch_100)"
    echo "Total jobs: 3"
    echo ""

    launch_test "baseline:1:False"
    launch_test "batch_10:10:True"
    launch_test "batch_100:100:True"
    ;;

  "batch_sizes")
    echo "Launching batch size comparison"
    echo "Total jobs: 4"
    echo ""

    launch_test "baseline:1:False"
    launch_test "batch_10:10:True"
    launch_test "batch_100:100:True"
    launch_test "batch_1000:1000:True"
    ;;

  "cache")
    echo "Launching cache effectiveness tests"
    echo "Total jobs: 3"
    echo ""

    launch_test "baseline:1:False"
    launch_test "cache_only:1:True"
    launch_test "batch_100:100:True"
    ;;

  "pool_size")
    echo "Launching pool size comparison"
    echo "Total jobs: 2"
    echo ""

    launch_test "batch_100:100:True"
    launch_test "small_pool_100:100:True:100"
    ;;

  "baseline")
    echo "Launching baseline only"
    echo ""

    launch_test "baseline:1:False"
    ;;

  "dry-run")
    echo "DRY RUN MODE - showing what would be launched"
    echo ""
    DRY_RUN=true

    for config in "${CONFIGS[@]}"; do
      launch_test "$config"
    done
    ;;

  "help" | "-h" | "--help")
    echo "Usage: $0 [mode] [OPTIONS]"
    echo ""
    echo "Modes:"
    echo "  all          - Launch all test configurations"
    echo "  quick        - Launch baseline + batch_10 + batch_100"
    echo "  batch_sizes  - Test different batch sizes (1, 10, 100, 1000)"
    echo "  cache        - Test cache effectiveness"
    echo "  pool_size    - Test pool size impact"
    echo "  baseline     - Launch baseline only"
    echo "  dry-run      - Show what would be launched without launching"
    echo "  help         - Show this help"
    echo ""
    echo "Environment Variables:"
    echo "  GPUS         - Number of GPUs per job (default: 1)"
    echo "  SPOT         - Use spot instances: '--spot' or '--no-spot' (default: --spot)"
    echo "  MAX_RUNTIME  - Max runtime in hours (default: 24)"
    echo "  GIT_REF      - Git branch/commit to use (default: current branch)"
    echo ""
    echo "Examples:"
    echo "  # Launch quick comparison with 2 GPUs, no spot"
    echo "  GPUS=2 SPOT=--no-spot $0 quick"
    echo ""
    echo "  # Launch all tests with 4-hour runtime"
    echo "  MAX_RUNTIME=4 $0 all"
    echo ""
    echo "  # Dry run to see what would be launched"
    echo "  $0 dry-run"
    echo ""
    echo "Available configurations:"
    for config in "${CONFIGS[@]}"; do
      IFS=':' read -r name batch_size cache pool_size <<< "$config"
      echo "  $name (batch=$batch_size, cache=$cache, pool=${pool_size:-1000})"
    done
    exit 0
    ;;

  *)
    echo "Unknown mode: $MODE"
    echo "Run '$0 help' for usage information"
    exit 1
    ;;
esac

echo ""
echo "=========================================="
echo "All launches complete!"
echo "=========================================="
echo ""
echo "To monitor jobs:"
echo "  uv run sky queue"
echo "  uv run sky status"
echo ""
echo "To check logs for a specific job:"
echo "  uv run sky logs <cluster-name>"
echo ""
echo "To download results:"
echo "  uv run sky down <cluster-name>  # After job completes"
echo ""
echo "Results will be in outputs/msb_perfdiagnosis_*/ directories"
echo ""
