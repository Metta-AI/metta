#!/bin/bash

# Function to monitor and benchmark a process
benchmark() {
  local process_type="$1"  # "train" or "replay"
  local command="$2"       # The command to run
  
  # Start timing
  start_time=$(date +%s.%N)
  
  # Run the command and capture its PID
  eval $command &
  CMD_PID=$!
  
  # Monitor memory in background
  (
    peak_memory=0
    while kill -0 $CMD_PID 2>/dev/null; do
      current_memory=$(ps -o rss= -p $CMD_PID | awk '{print $1/1024}')
      if (( $(echo "$current_memory > $peak_memory" | bc -l) )); then
        peak_memory=$current_memory
      fi
      sleep 0.5
    done
    echo $peak_memory > /tmp/${process_type}_peak_memory.txt
  ) &
  
  # Wait for the main process to complete
  wait $CMD_PID
  exit_code=$?
  
  # End timing
  end_time=$(date +%s.%N)
  duration=$(echo "$end_time - $start_time" | bc)
  
  # Get the peak memory
  if [ -f "/tmp/${process_type}_peak_memory.txt" ]; then
    peak_memory=$(cat /tmp/${process_type}_peak_memory.txt)
  else
    echo "Warning: Could not measure memory usage for ${process_type}, using fallback value"
    peak_memory=0
  fi
  
  # Output the results
  echo "duration=$duration" >> $GITHUB_OUTPUT
  echo "memory_usage=$peak_memory" >> $GITHUB_OUTPUT
  echo "${process_type^} duration: $duration seconds"
  echo "${process_type^} memory: $peak_memory MB"
  
  return $exit_code
}