#!/bin/bash
# Test matrix for LP performance optimizations
# Tests different combinations of settings to find optimal configuration

set -e

echo "=========================================="
echo "LP Performance Optimization Test Matrix"
echo "=========================================="
echo ""

# Test configurations
declare -A CONFIGS=(
  ["baseline"]="perf_invalidation_batch_size=1 perf_cache_task_list=False perf_log_metrics=True"
  ["batch_10"]="perf_invalidation_batch_size=10 perf_cache_task_list=True perf_log_metrics=True"
  ["batch_100"]="perf_invalidation_batch_size=100 perf_cache_task_list=True perf_log_metrics=True"
  ["batch_1000"]="perf_invalidation_batch_size=1000 perf_cache_task_list=True perf_log_metrics=True"
  ["cache_only"]="perf_invalidation_batch_size=1 perf_cache_task_list=True perf_log_metrics=True"
  ["small_pool_100"]="perf_invalidation_batch_size=100 perf_cache_task_list=True num_active_tasks=100 perf_log_metrics=True"
)

# Timeout for each test (10 minutes = 600 seconds)
TIMEOUT=600

# Function to run a test
run_perf_test() {
  local test_name=$1
  local config_params=$2

  echo ""
  echo "===================================="
  echo "Test: $test_name"
  echo "===================================="
  echo "Config: $config_params"
  echo "Timeout: ${TIMEOUT}s"
  echo ""

  # Run the test
  timeout ${TIMEOUT}s uv run ./tools/run.py cogs_v_clips.train \
    use_lp=True \
    run=msb_perfdiagnosis_${test_name} \
    ${config_params} || true

  echo ""
  echo "Test complete: $test_name"
  echo ""
}

# Parse arguments
MODE=${1:-all}

case "$MODE" in
  "all")
    echo "Running all performance tests"
    echo "Total estimated time: ~$((${#CONFIGS[@]} * TIMEOUT / 60)) minutes"
    echo ""

    for test_name in "${!CONFIGS[@]}"; do
      run_perf_test "$test_name" "${CONFIGS[$test_name]}"
    done
    ;;

  "quick")
    echo "Running quick comparison (baseline, batch_10, batch_100)"
    echo ""

    run_perf_test "baseline" "${CONFIGS[baseline]}"
    run_perf_test "batch_10" "${CONFIGS[batch_10]}"
    run_perf_test "batch_100" "${CONFIGS[batch_100]}"
    ;;

  "baseline")
    run_perf_test "baseline" "${CONFIGS[baseline]}"
    ;;

  "batch_sizes")
    echo "Testing different batch sizes"
    echo ""

    run_perf_test "baseline" "${CONFIGS[baseline]}"
    run_perf_test "batch_10" "${CONFIGS[batch_10]}"
    run_perf_test "batch_100" "${CONFIGS[batch_100]}"
    run_perf_test "batch_1000" "${CONFIGS[batch_1000]}"
    ;;

  "cache")
    echo "Testing cache effectiveness"
    echo ""

    run_perf_test "baseline" "${CONFIGS[baseline]}"
    run_perf_test "cache_only" "${CONFIGS[cache_only]}"
    run_perf_test "batch_100" "${CONFIGS[batch_100]}"
    ;;

  "pool_size")
    echo "Testing pool size impact"
    echo ""

    run_perf_test "batch_100" "${CONFIGS[batch_100]}"
    run_perf_test "small_pool_100" "${CONFIGS[small_pool_100]}"
    ;;

  *)
    # Run a specific test
    if [[ -n "${CONFIGS[$MODE]}" ]]; then
      run_perf_test "$MODE" "${CONFIGS[$MODE]}"
    else
      echo "Unknown test: $MODE"
      echo ""
      echo "Usage: $0 [mode]"
      echo ""
      echo "Modes:"
      echo "  all          - Run all tests"
      echo "  quick        - Run baseline + batch_10 + batch_100"
      echo "  batch_sizes  - Test different batch sizes (1, 10, 100, 1000)"
      echo "  cache        - Test cache effectiveness"
      echo "  pool_size    - Test pool size impact"
      echo "  baseline     - Run baseline only"
      echo ""
      echo "Available tests:"
      for test_name in "${!CONFIGS[@]}"; do
        echo "  $test_name"
      done
      exit 1
    fi
    ;;
esac

echo ""
echo "=========================================="
echo "All tests complete!"
echo "=========================================="
echo ""
echo "To compare results, run:"
echo "  ./tools/compare_lp_performance.sh"
echo ""
