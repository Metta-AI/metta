#!/bin/bash
# Compare LP performance test results

echo "=========================================="
echo "LP Performance Test Comparison"
echo "=========================================="
echo ""

# Test names to compare
TESTS=("baseline" "batch_10" "batch_100" "batch_1000" "cache_only" "small_pool_100")

# Function to extract SPS values from logs
extract_sps() {
  local test_name=$1
  local log_file=$(ls -t outputs/*msb_perfdiagnosis_${test_name}*/logs/train.log 2> /dev/null | head -1)

  if [ -z "$log_file" ]; then
    echo "N/A"
    return
  fi

  # Extract SPS values
  local sps_values=$(grep -iE "sps:|steps_per_second" "$log_file" \
    | grep -oE "[0-9]+\.[0-9]+" | head -20)

  if [ -z "$sps_values" ]; then
    echo "N/A"
    return
  fi

  # Calculate stats
  local count=$(echo "$sps_values" | wc -l)
  local first=$(echo "$sps_values" | head -1)
  local last=$(echo "$sps_values" | tail -1)
  local avg=$(echo "$sps_values" | awk '{sum+=$1} END {print sum/NR}')

  # Calculate degradation
  if [ "$first" != "0" ]; then
    local degradation=$(echo "scale=1; (($first - $last) / $first) * 100" | bc)
  else
    local degradation="N/A"
  fi

  echo "$first|$last|$avg|$degradation|$count"
}

# Print table header
printf "%-20s %12s %12s %12s %15s %10s\n" \
  "Configuration" "Initial SPS" "Final SPS" "Avg SPS" "Degradation %" "Samples"
echo "--------------------------------------------------------------------------------"

# Collect and display results
for test in "${TESTS[@]}"; do
  result=$(extract_sps "$test")

  if [ "$result" == "N/A" ]; then
    printf "%-20s %12s %12s %12s %15s %10s\n" "$test" "N/A" "N/A" "N/A" "N/A" "N/A"
    continue
  fi

  IFS='|' read -r first last avg degradation count <<< "$result"

  printf "%-20s %12.1f %12.1f %12.1f %14s%% %10s\n" \
    "$test" "$first" "$last" "$avg" "$degradation" "$count"
done

echo ""
echo "=========================================="
echo "Performance Metrics"
echo "=========================================="
echo ""

# Extract performance metrics from logs
for test in "${TESTS[@]}"; do
  log_file=$(ls -t outputs/*msb_perfdiagnosis_${test}*/logs/train.log 2> /dev/null | head -1)

  if [ -z "$log_file" ]; then
    continue
  fi

  echo "Test: $test"
  echo "----------------------------------------"

  # Extract invalidation stats
  invalidations=$(grep "\[LP_PERF\] Invalidations" "$log_file" | tail -1)
  if [ -n "$invalidations" ]; then
    echo "$invalidations"
  fi

  # Extract cache stats
  cache_stats=$(grep "\[LP_PERF\] Task list cache" "$log_file" | tail -1)
  if [ -n "$cache_stats" ]; then
    echo "$cache_stats"
  fi

  echo ""
done

echo ""
echo "=========================================="
echo "Recommendations"
echo "=========================================="
echo ""

# Simple recommendation logic
best_test=""
best_avg=0

for test in "${TESTS[@]}"; do
  result=$(extract_sps "$test")

  if [ "$result" == "N/A" ]; then
    continue
  fi

  IFS='|' read -r first last avg degradation count <<< "$result"

  if (($(echo "$avg > $best_avg" | bc -l))); then
    best_avg=$avg
    best_test=$test
  fi
done

if [ -n "$best_test" ]; then
  echo "Best performing configuration: $best_test"
  echo "Average SPS: $best_avg"
  echo ""

  # Get the config params
  if grep -q "perf_invalidation_batch_size" outputs/*msb_perfdiagnosis_${best_test}*/config.yaml 2> /dev/null; then
    echo "Configuration details:"
    grep "perf_" outputs/*msb_perfdiagnosis_${best_test}*/config.yaml 2> /dev/null || echo "  (config not found)"
  fi
else
  echo "No test results found. Run tests first with:"
  echo "  ./tools/test_lp_performance_matrix.sh quick"
fi

echo ""
