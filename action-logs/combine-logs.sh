#!/bin/zsh

# Combine all logs with metadata headers
for dir in action-logs/*/; do
  if [[ ! -d "$dir" ]]; then
    continue
  fi
  
  dir_name=$(basename "$dir")
  run_id=${dir_name%%-*}
  
  echo "Processing directory: $dir (run_id: $run_id)"
  
  # Get run info from JSON
  run_info=$(jq -r ".[] | select(.databaseId == $run_id) | \"Workflow: \(.workflowName) | Run: \(.name) | Branch: \(.headBranch) | Created: \(.createdAt)\"" runs.json 2>/dev/null)
  
  echo "=== RUN ID: $run_id ===" >> combined-logs.txt
  echo "$run_info" >> combined-logs.txt
  echo "========================" >> combined-logs.txt
  
  # Find all text files and process them
  for logfile in "$dir"/**/*.txt(N); do
    if [[ -f "$logfile" ]]; then
      echo "Processing log file: $logfile"
      echo "--- $(basename "$logfile") ---" >> combined-logs.txt
      cat "$logfile" >> combined-logs.txt
      echo "" >> combined-logs.txt
    fi
  done
  
  # Also check for files without .txt extension that might be logs
  for logfile in "$dir"/**/*(N); do
    if [[ -f "$logfile" && ! "$logfile" =~ "\.(zip|json)$" ]]; then
      # Check if it's a text file
      if file "$logfile" | grep -q "text"; then
        echo "Processing text file: $logfile"
        echo "--- $(basename "$logfile") ---" >> combined-logs.txt
        cat "$logfile" >> combined-logs.txt
        echo "" >> combined-logs.txt
      fi
    fi
  done
  
  echo "" >> combined-logs.txt
done

echo "Done! Check combined-logs.txt"
