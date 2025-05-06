#!/bin/bash

# Benchmark function for GitHub Actions
# Measures execution time and peak memory usage

benchmark() {
  local process_type="$1" # "train" or "replay"
  local command="$2"      # The command to run
  local temp_file="/tmp/${process_type}_peak_memory.txt"

  # Clean up any existing temp file
  rm -f "$temp_file"

  echo "Starting benchmark for: $process_type"

  # Start timing
  start_time=$(date +%s.%N)

  # Run the command and capture its PID
  eval $command &
  CMD_PID=$!

  # Monitor memory in background
  (
    peak_memory=0
    while ps -p $CMD_PID > /dev/null 2>&1; do
      # Get memory usage (VmRSS from /proc is more reliable)
      if [ -f "/proc/$CMD_PID/status" ]; then
        current_memory=$(grep VmRSS /proc/$CMD_PID/status | awk '{print $2}')
        # Convert kB to MB
        current_memory=$(echo "scale=2; ${current_memory:-0} / 1024" | bc)
      else
        # Fallback to ps if /proc is not available
        current_memory=$(ps -o rss= -p $CMD_PID | awk '{print $1 / 1024}')
      fi

      # Update peak value if current is higher
      if (($(echo "$current_memory > $peak_memory" | bc -l))); then
        peak_memory=$current_memory
      fi

      sleep 0.1
    done

    # Save peak memory
    echo "$peak_memory" > "$temp_file"
  ) &
  MONITOR_PID=$!

  # Wait for the main process to complete
  wait $CMD_PID
  exit_code=$?

  # Allow monitor process to finish writing
  sleep 0.2

  # Terminate the monitoring process
  kill $MONITOR_PID 2> /dev/null || true

  # End timing
  end_time=$(date +%s.%N)
  duration=$(echo "$end_time - $start_time" | bc)

  # Get the peak memory
  peak_memory=0
  if [ -f "$temp_file" ]; then
    peak_memory=$(cat "$temp_file")
    # Validate numeric value
    if ! [[ "$peak_memory" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
      echo "Warning: Invalid memory measurement, using fallback value"
      peak_memory=0
    fi
  else
    echo "Warning: Could not measure memory usage for ${process_type}, using fallback value"
  fi

  # Clean up
  rm -f "$temp_file"

  # Output results for GitHub Actions
  echo "duration=$duration" >> $GITHUB_OUTPUT
  echo "memory_usage=$peak_memory" >> $GITHUB_OUTPUT

  # Output for human readability
  echo "${process_type^} duration: $duration seconds"
  echo "${process_type^} memory: $peak_memory MB"

  return $exit_code
}

# Export the function so it's available in subshells
export -f benchmark

# Example usage:
# benchmark "train" "python train.py --epochs 10"
