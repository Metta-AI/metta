#!/bin/bash
set -e # Exit immediately if any command fails

# How to use:
# 1. First, run the setup to save your current working files:
#    ```bash
#    ./devops/tools/git-history-trainer.sh --setup
#    ```
#
# 2. Then run the script to process multiple commits in parallel:
#    ```bash
#    ./devops/tools/git-history-trainer.sh --commits=3  --commit-interval=5 # Process (1, 6, 11) commits back from HEAD
#    ```
#
# 3. This script makes a lot of branches. You can clean up this way:
#    ```bash
#    ./devops/tools/git-history-trainer.sh --cleanup
#    ```
#
# This approach can be more efficient than the traditional git bisect for our use case, since:
#
# - It allows parallel testing of multiple commits at once
# - Each job runs independently, so you can see which ones fail and which ones succeed
# - The comprehensive log file will help you track all the jobs and their details
# - The timestamp in the run ID will make it easy to sort and identify jobs chronologically
#
# When all jobs are complete, you can review the results and look for the transition point where jobs
# start failing - that should identify the commit that introduced the issue.

# turn off git warnings
git config advice.ignoredHook false
git config advice.detachedHead false

# Configuration
START_COMMIT="HEAD"
NUM_COMMITS=10    # How many commits back to process
COMMIT_INTERVAL=1 # Default: process every commit
USER=$(whoami)
TMP_STORAGE="/tmp/git_debug_files_$USER"
TIMESTAMP=$(date +%s)
LOG_FILE="/tmp/debug_log_$TIMESTAMP.txt"
SUMMARY_FILE="/tmp/debug_summary_$TIMESTAMP.txt"

# Initialize summary file with header
echo "RUN_ID,COMMIT,COMMIT_MESSAGE,COMMIT_DATETIME" > "$SUMMARY_FILE"

# Function to show usage
show_usage() {
  echo "Usage: $0 [--setup|--commits=N|--cleanup|--commit-interval=N|--skip-commits=N]"
  echo "  --setup             Save current working files to temp storage"
  echo "  --commits=N         Process N commits back from HEAD (default: 10)"
  echo "  --skip-commits=N    Skip N most recent commits before processing (default: 0)"
  echo "  --commit-interval=N Skip N-1 commits between each processed commit (default: 1)"
  echo "  --cleanup           Remove all local and remote debug-test branches"
  echo "  (no args)           Apply saved files, create test branches, and launch jobs"
  exit 1
}

# Parse arguments
SKIP_COMMITS=0 # Default: start from HEAD
for arg in "$@"; do
  case $arg in
    --setup)
      SETUP_MODE=true
      ;;
    --commits=*)
      NUM_COMMITS="${arg#*=}"
      ;;
    --skip-commits=*)
      SKIP_COMMITS="${arg#*=}"
      ;;
    --commit-interval=*)
      COMMIT_INTERVAL="${arg#*=}"
      ;;
    --cleanup)
      CLEANUP_MODE=true
      ;;
    --help)
      show_usage
      ;;
  esac
done

# Handle cleanup mode
if [ "$CLEANUP_MODE" = true ]; then
  echo "Cleaning up debug test branches and temporary files..."

  # Clean local branches
  echo "Deleting local debug-test branches:"
  git branch | grep "debug-test" | xargs git branch -D 2> /dev/null || echo "No local branches to delete"

  # Clean remote branches
  echo "Deleting remote debug-test branches:"
  remote_branches=$(git ls-remote --heads origin | grep "debug-test" | awk '{print $2}' | sed 's/refs\/heads\///')
  if [ -n "$remote_branches" ]; then
    for branch in $remote_branches; do
      echo "  Deleting: $branch"
      git push origin --delete "$branch"
    done
  else
    echo "No remote branches to delete"
  fi

  # Clean up temporary files
  if [ -d "$TMP_STORAGE" ]; then
    echo "Removing temporary storage directory: $TMP_STORAGE"
    rm -rf "$TMP_STORAGE"
  else
    echo "No temporary storage directory found at $TMP_STORAGE"
  fi

  # Clean up log files
  echo "Cleaning up debug log files..."
  debug_logs=$(find /tmp -name "debug_log_*.txt" 2> /dev/null)
  if [ -n "$debug_logs" ]; then
    echo "Found log files:"
    echo "$debug_logs"
    echo "Removing log files..."
    rm -f /tmp/debug_log_*.txt
  fi

  # Clean up summary files
  debug_summaries=$(find /tmp -name "debug_summary_*.txt" 2> /dev/null)
  if [ -n "$debug_summaries" ]; then
    echo "Found summary files:"
    echo "$debug_summaries"
    echo "Removing summary files..."
    rm -f /tmp/debug_summary_*.txt
  fi

  echo "Cleanup complete!"
  exit 0
fi

# Setup mode - copy the current working files to temporary storage
if [ "$SETUP_MODE" = true ]; then
  echo "Setting up temporary storage at $TMP_STORAGE"

  # Create the storage directory
  mkdir -p "$TMP_STORAGE"

  # Copy the specified files and directories
  echo "Copying files to temporary storage..."

  # Copy directories
  cp -r "$(git rev-parse --show-toplevel)/devops" "$TMP_STORAGE/"

  # Create subdirectories as needed
  mkdir -p "$TMP_STORAGE/mettagrid"
  mkdir -p "$TMP_STORAGE/metta/util" # Add directory for git utilities

  # Copy individual files
  cp "$(git rev-parse --show-toplevel)/mettagrid/pyproject.toml" "$TMP_STORAGE/mettagrid/"
  cp "$(git rev-parse --show-toplevel)/mettagrid/Makefile" "$TMP_STORAGE/mettagrid/"
  cp "$(git rev-parse --show-toplevel)/mettagrid/setup.py" "$TMP_STORAGE/mettagrid/"
  cp "$(git rev-parse --show-toplevel)/Makefile" "$TMP_STORAGE/"
  cp "$(git rev-parse --show-toplevel)/pyproject.toml" "$TMP_STORAGE/"
  cp "$(git rev-parse --show-toplevel)/uv.lock" "$TMP_STORAGE/"
  cp "$(git rev-parse --show-toplevel)/setup.py" "$TMP_STORAGE/"

  # Copy git utilities module
  echo "Copying git utilities module..."
  cp "$(git rev-parse --show-toplevel)/metta/util/git.py" "$TMP_STORAGE/metta/util/"
  cp "$(git rev-parse --show-toplevel)/metta/util/__init__.py" "$TMP_STORAGE/metta/util/" 2> /dev/null || touch "$TMP_STORAGE/metta/util/__init__.py"

  # Create the patch for experience.py
  echo "Creating experience.py patch file"
  cat > "$TMP_STORAGE/experience_patch.py" << 'EOL'
    def flatten_batch(self, advantages_np: np.ndarray) -> None:
        advantages: torch.Tensor = torch.as_tensor(advantages_np).to(self.device, non_blocking=True)

        if self.b_idxs_obs is None:
            raise ValueError("b_idxs_obs is None - call sort_training_data first")

        b_idxs, b_flat = self.b_idxs, self.b_idxs_flat

        # Process the batch data
        self.b_actions = self.actions.to(self.device, non_blocking=True, dtype=torch.long)
        self.b_logprobs = self.logprobs.to(self.device, non_blocking=True)
        self.b_dones = self.dones.to(self.device, non_blocking=True)
        self.b_values = self.values.to(self.device, non_blocking=True)

        # Reshape advantages for minibatches
        self.b_advantages = (
            advantages.reshape(self.minibatch_rows, self.num_minibatches, self.bptt_horizon)
            .transpose(0, 1)
            .reshape(self.num_minibatches, self.minibatch_size)
        )

        self.returns_np = advantages_np + self.values_np

        # Process the rest of the batch data
        self.b_obs = self.obs[self.b_idxs_obs]
        self.b_actions = self.b_actions[b_idxs].contiguous()
        self.b_logprobs = self.b_logprobs[b_idxs]
        self.b_dones = self.b_dones[b_idxs]
        self.b_values = self.b_values[b_flat]
        self.b_returns = self.b_advantages + self.b_values
EOL

  echo "Setup complete. Files saved to $TMP_STORAGE"

  exit 0
fi

# Check if temporary storage exists
if [ ! -d "$TMP_STORAGE" ]; then
  echo "Error: Temporary storage not found at $TMP_STORAGE"
  echo "Please run '$0 --setup' first to save your working files"
  exit 1
fi

# Create a log file to track jobs
echo "Starting debug job launcher at $(date)" > "$LOG_FILE"
echo "Will process $NUM_COMMITS commits starting from $START_COMMIT" >> "$LOG_FILE"
if [ "$SKIP_COMMITS" -gt 0 ]; then
  echo "Skipping $SKIP_COMMITS most recent commits" >> "$LOG_FILE"
fi
echo "============================================================" >> "$LOG_FILE"

# Save the current branch so we can return to it at the end
ORIGINAL_BRANCH=$(git symbolic-ref --quiet HEAD || git rev-parse HEAD)
if [[ $ORIGINAL_BRANCH == refs/heads/* ]]; then
  ORIGINAL_BRANCH=${ORIGINAL_BRANCH#refs/heads/}
fi
echo "Original branch: $ORIGINAL_BRANCH" >> "$LOG_FILE"

# Get list of commits to process based on the interval
if [ "$SKIP_COMMITS" -gt 0 ]; then
  echo "Skipping $SKIP_COMMITS most recent commits"
  # Adjust the START_COMMIT to skip the specified number of commits
  START_COMMIT="HEAD~$SKIP_COMMITS"
else
  START_COMMIT="HEAD"
fi

if [ "$COMMIT_INTERVAL" -gt 1 ]; then
  echo "Using commit interval of $COMMIT_INTERVAL (processing every ${COMMIT_INTERVAL}th commit)"

  # Get total available commits
  ALL_COMMITS=$(git rev-list $START_COMMIT)
  TOTAL_COMMITS=$(echo "$ALL_COMMITS" | wc -l | tr -d ' ')

  # Calculate the total depth we need to go
  MAX_DEPTH=$((NUM_COMMITS * COMMIT_INTERVAL))
  TOTAL_DEPTH=$((SKIP_COMMITS + MAX_DEPTH))

  # Make sure we don't try to go deeper than the repo history
  if [ "$TOTAL_DEPTH" -gt "$TOTAL_COMMITS" ]; then
    echo "Warning: Requested depth ($TOTAL_DEPTH) exceeds repository history ($TOTAL_COMMITS)"
    echo "Will process up to the oldest available commit"
  fi

  # Select commits at specified intervals
  COMMITS=""
  for i in $(seq 0 $((NUM_COMMITS - 1))); do
    DEPTH=$((i * COMMIT_INTERVAL))
    if [ "$DEPTH" -lt "$TOTAL_COMMITS" ]; then
      NEW_COMMIT=$(git rev-parse $START_COMMIT~$DEPTH)
      COMMITS="$COMMITS $NEW_COMMIT"
    fi
  done
else
  # Default: get consecutive commits
  COMMITS=$(git rev-list --max-count=$NUM_COMMITS $START_COMMIT)
fi

# Count actual commits found
COMMIT_COUNT=$(echo "$COMMITS" | wc -l | tr -d ' ')
echo "Found $COMMIT_COUNT commits to process"

# Start processing commits
echo "Starting to process $COMMIT_COUNT commits..."
COUNTER=1
JOBS_SUMMARY=""

for COMMIT in $COMMITS; do
  COMMIT_SHORT=$(git rev-parse --short "$COMMIT")
  COMMIT_DATE=$(git show -s --format=%ci "$COMMIT")
  COMMIT_MSG=$(git show -s --format=%s "$COMMIT")

  echo ""
  echo "============================================================"
  echo "Processing commit $COUNTER of $COMMIT_COUNT: $COMMIT_SHORT"
  echo "Date: $COMMIT_DATE"
  echo "Message: $COMMIT_MSG"
  echo "============================================================"

  # Log the commit info
  echo "" >> "$LOG_FILE"
  echo "Commit $COUNTER of $COMMIT_COUNT: $COMMIT_SHORT" >> "$LOG_FILE"
  echo "Date: $COMMIT_DATE" >> "$LOG_FILE"
  echo "Message: $COMMIT_MSG" >> "$LOG_FILE"

  # Clean up any untracked files that might interfere with checkout
  echo "Cleaning up untracked files from previous run..."
  REPO_ROOT=$(git rev-parse --show-toplevel)
  git clean -fd "$REPO_ROOT/devops"

  # Check out this commit
  git checkout "$COMMIT" -f

  # Create a unique test branch
  TIMESTAMP=$(date +%s)
  TEST_BRANCH="debug-test-rwalters-$COMMIT_SHORT-$TIMESTAMP"
  echo "Creating test branch: $TEST_BRANCH"
  git checkout -b "$TEST_BRANCH"

  # Process this commit
  REPO_ROOT=$(git rev-parse --show-toplevel)

  # Create directory structure first
  mkdir -p "$REPO_ROOT/metta/util"
  mkdir -p "$REPO_ROOT/mettagrid"
  mkdir -p "$REPO_ROOT/devops"

  # Copy directories with -f to force overwrite
  echo "Copying devops directory..."
  cp -rf "$TMP_STORAGE/devops"/* "$REPO_ROOT/devops/" 2> /dev/null || :

  # Copy git utilities module
  echo "Restoring git utilities module..."
  cp -f "$TMP_STORAGE/metta/util/git.py" "$REPO_ROOT/metta/util/"
  cp -f "$TMP_STORAGE/metta/util/__init__.py" "$REPO_ROOT/metta/util/"

  # Copy individual files with -f to force overwrite
  echo "Copying individual files..."
  cp -f "$TMP_STORAGE/mettagrid/pyproject.toml" "$REPO_ROOT/mettagrid/"
  cp -f "$TMP_STORAGE/mettagrid/Makefile" "$REPO_ROOT/mettagrid/"
  cp -f "$TMP_STORAGE/mettagrid/setup.py" "$REPO_ROOT/mettagrid/"
  cp -f "$TMP_STORAGE/Makefile" "$REPO_ROOT/"
  cp -f "$TMP_STORAGE/pyproject.toml" "$REPO_ROOT/"
  cp -f "$TMP_STORAGE/uv.lock" "$REPO_ROOT/"
  cp -f "$TMP_STORAGE/setup.py" "$REPO_ROOT/"

  # Check if experience.py exists and patch it
  EXPERIENCE_PATH="$REPO_ROOT/metta/rl/pufferlib/experience.py"
  if [ -f "$EXPERIENCE_PATH" ]; then
    echo "Patching metta/rl/pufferlib/experience.py"

    # Simple approach: truncate the file at "def flatten_batch" and append our implementation
    # First find the line number where flatten_batch starts
    FLATTEN_LINE=$(grep -n "    def flatten_batch" "$EXPERIENCE_PATH" | head -1 | cut -d: -f1)

    if [ -n "$FLATTEN_LINE" ]; then
      echo "Found flatten_batch at line $FLATTEN_LINE - replacing it..."

      # Keep everything before flatten_batch
      head -n $((FLATTEN_LINE - 1)) "$EXPERIENCE_PATH" > "${EXPERIENCE_PATH}.new"

      # Add our implementation from the patch file
      cat "$TMP_STORAGE/experience_patch.py" >> "${EXPERIENCE_PATH}.new"

      # Replace the original file
      mv "${EXPERIENCE_PATH}.new" "$EXPERIENCE_PATH"
      echo "Successfully patched experience.py"
    else
      echo "Could not find flatten_batch function in experience.py"
      echo "Adding our implementation at the end of the file..."

      # Append our implementation to the end of the file
      echo "" >> "$EXPERIENCE_PATH"
      echo "    def flatten_batch" >> "$EXPERIENCE_PATH"
      cat "$TMP_STORAGE/experience_patch.py" >> "$EXPERIENCE_PATH"

      echo "Added our implementation at the end of experience.py"
    fi
  else
    echo "Warning: experience.py doesn't exist in commit $COMMIT_SHORT. Skipping..."
    echo "experience.py NOT FOUND in commit $COMMIT_SHORT. Skipped." >> "$LOG_FILE"
  fi

  # Add a marker file to force a git change
  echo "# debug test marker - testing commit $COMMIT_SHORT" > "$REPO_ROOT/DEBUG_TEST_MARKER"

  # Commit the changes
  git add -f "$REPO_ROOT/DEBUG_TEST_MARKER"
  git add -f "$REPO_ROOT/metta/util/git.py"
  git add -f "$REPO_ROOT/devops"
  git add -f "$REPO_ROOT/metta/rl/pufferlib/experience.py"
  git add -f "$REPO_ROOT/mettagrid"
  git add -f "$REPO_ROOT/Makefile"
  git add -f "$REPO_ROOT/pyproject.toml"
  git add -f "$REPO_ROOT/uv.lock"
  git add -f "$REPO_ROOT/setup.py"

  echo "Committing changes to test branch"
  git commit -m "debug test: Applied fixes for testing commit $COMMIT_SHORT" || echo "No changes to commit"

  # Push the branch to remote
  echo "Pushing test branch to remote"
  git push -f origin "$TEST_BRANCH"

  # Create a unique run name for this job
  JOB_TIMESTAMP=$(date +%s)
  RUN_NAME="d.${USER}.${COMMIT_SHORT}.${JOB_TIMESTAMP}"

  # Submit the job using your launch_task script
  echo "Submitting job for commit $COMMIT_SHORT using launch_task.py..."
  python -m devops.aws.batch.launch_task \
    --cmd=train \
    --run="$RUN_NAME" \
    --git-branch="$TEST_BRANCH" \
    --timeout-minutes=30 \
    --skip-validation \
    --skip-push-check \
    --force \
    trainer.env=env/mettagrid/arena/advanced

  # Log job details
  echo "Job submitted for commit: $COMMIT_SHORT" | tee -a "$LOG_FILE"
  echo "  Branch: $TEST_BRANCH" | tee -a "$LOG_FILE"
  echo "  Run ID: $RUN_NAME" | tee -a "$LOG_FILE"

  # Properly escape the commit message for CSV format
  COMMIT_MSG_ESCAPED=$(echo "$COMMIT_MSG" | sed 's/"/"""/g')

  # Add to summary file for the final table
  echo "$RUN_NAME,$COMMIT_SHORT,\"$COMMIT_MSG_ESCAPED\",$COMMIT_DATE" >> "$SUMMARY_FILE"

  COUNTER=$((COUNTER + 1))

  # Clean up any Python cache files or other untracked files before moving to the next commit
  echo "Cleaning up before moving to the next commit..."
  find "$REPO_ROOT" -name "__pycache__" -type d -exec rm -rf {} + 2> /dev/null || :
  find "$REPO_ROOT" -name "*.pyc" -delete 2> /dev/null || :
  find "$REPO_ROOT" -name "*.pyo" -delete 2> /dev/null || :
  find "$REPO_ROOT" -name ".DS_Store" -delete 2> /dev/null || :

  # Force clean untracked files
  git clean -fd
done

# Return to the original branch
echo "Returning to original branch: $ORIGINAL_BRANCH"
git checkout "$ORIGINAL_BRANCH"

# Define column widths for the table
COL_WIDTH_RUNID=40
COL_WIDTH_COMMIT=12
COL_WIDTH_MSG=40
COL_WIDTH_DATE=30
TOTAL_WIDTH=$((COL_WIDTH_RUNID + COL_WIDTH_COMMIT + COL_WIDTH_MSG + COL_WIDTH_DATE + 3 * 3))

# Print the summary table of submitted jobs
echo ""
echo "$(printf '%0.s=' $(seq 1 $TOTAL_WIDTH))"
echo "$(printf '%*s' $(((TOTAL_WIDTH + 23) / 2)) "SUMMARY OF SUBMITTED JOBS")"
echo "$(printf '%0.s=' $(seq 1 $TOTAL_WIDTH))"
echo ""

# Format and print the table using column for better formatting
if [ -f "$SUMMARY_FILE" ]; then
  # Create a temporary file with formatted header
  printf "%-${COL_WIDTH_RUNID}s | %-${COL_WIDTH_COMMIT}s | %-${COL_WIDTH_MSG}s | %-${COL_WIDTH_DATE}s\n" \
    "RUN_ID" "COMMIT" "COMMIT_MESSAGE" "COMMIT_DATETIME" > /tmp/header.txt

  # Create separator line with correct lengths
  printf "%s | %s | %s | %s\n" \
    "$(printf '%0.s-' $(seq 1 $COL_WIDTH_RUNID))" \
    "$(printf '%0.s-' $(seq 1 $COL_WIDTH_COMMIT))" \
    "$(printf '%0.s-' $(seq 1 $COL_WIDTH_MSG))" \
    "$(printf '%0.s-' $(seq 1 $COL_WIDTH_DATE))" >> /tmp/header.txt

  # Merge the header and data
  cat /tmp/header.txt > /tmp/formatted.txt

  # Process CSV using a more portable approach without gensub
  # This avoids the GNU-specific gensub function
  tail -n +2 "$SUMMARY_FILE" | while IFS=, read -r run_id commit rest; do
    # Extract the commit message which might have commas inside quotes
    if [[ "$rest" == \"* ]]; then
      # The commit message is in quotes
      commit_msg=$(echo "$rest" | sed -E 's/^"([^"]*)"(,.*)$/\1/')
      commit_date=$(echo "$rest" | sed -E 's/^"[^"]*"(,.*)$/\1/' | sed 's/^,//')
    else
      # No quotes in the commit message
      commit_msg=$(echo "$rest" | cut -d, -f1)
      commit_date=$(echo "$rest" | cut -d, -f2-)
    fi

    # Truncate message if too long
    if [ ${#commit_msg} -gt $((COL_WIDTH_MSG - 3)) ]; then
      commit_msg="${commit_msg:0:$((COL_WIDTH_MSG - 3))}..."
    fi

    # Print formatted row
    printf "%-${COL_WIDTH_RUNID}s | %-${COL_WIDTH_COMMIT}s | %-${COL_WIDTH_MSG}s | %s\n" \
      "$run_id" "$commit" "$commit_msg" "$commit_date" >> /tmp/formatted.txt
  done

  # Print the formatted table
  cat /tmp/formatted.txt

  # Clean up temporary files
  rm -f /tmp/header.txt /tmp/formatted.txt

  echo ""
  echo "Full details saved to: $SUMMARY_FILE"
else
  echo "No jobs were submitted."
fi

echo ""
echo "============================================================"
echo "Debug job submission complete!"
echo "Processed $COMMIT_COUNT commits."
echo "Log file: $LOG_FILE"
echo "Summary file: $SUMMARY_FILE"
echo "============================================================"
echo ""
echo "Review the log file to see all the jobs that were submitted."
exit 0
