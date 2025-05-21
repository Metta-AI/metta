#!/bin/bash
set -e  # Exit immediately if any command fails

# How to use:
# 1. First, run the setup to save your current working files:
#    ```bash
#    ./debug-helper.sh --setup
#    ```

# 2. Then run the script to process multiple commits in parallel:
#    ```bash
#    ./debug-helper.sh --commits=20  # Process 20 commits back from HEAD
#    ```

# This approach is more efficient than the traditional git bisect for our use case, since:

# 1. It allows parallel testing of multiple commits at once
# 2. Each job runs independently, so you can see which ones fail and which ones succeed
# 3. The comprehensive log file will help you track all the jobs and their details
# 4. The timestamp in the run ID will make it easy to sort and identify jobs chronologically

# When all jobs are complete, you can review the results and look for the transition point where jobs 
# start failing - that should identify the commit that introduced the issue.

# Quiet git hook warnings
git config advice.ignoredHook false
# Set detached HEAD advice to false
git config advice.detachedHead false

# Configuration
START_COMMIT="HEAD"
NUM_COMMITS=10  # How many commits back to process
USER=$(whoami)
TMP_STORAGE="/tmp/git_debug_files_$USER"
TIMESTAMP=$(date +%s)
LOG_FILE="/tmp/debug_log_$TIMESTAMP.txt"

# Function to show usage
show_usage() {
  echo "Usage: $0 [--setup] [--commits=N]"
  echo "  --setup         Save current working files to temp storage"
  echo "  --commits=N     Process N commits back from HEAD (default: 10)"
  echo "  (no args)       Apply saved files, create test branches, and launch jobs"
  exit 1
}

# Parse arguments
for arg in "$@"; do
  case $arg in
    --setup)
      SETUP_MODE=true
      ;;
    --commits=*)
      NUM_COMMITS="${arg#*=}"
      ;;
    --help)
      show_usage
      ;;
  esac
done

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
  mkdir -p "$TMP_STORAGE/metta/util"  # Add directory for git utilities
  
  # Copy individual files
  cp "$(git rev-parse --show-toplevel)/mettagrid/pyproject.toml" "$TMP_STORAGE/mettagrid/"
  cp "$(git rev-parse --show-toplevel)/mettagrid/Makefile" "$TMP_STORAGE/mettagrid/"
  cp "$(git rev-parse --show-toplevel)/mettagrid/setup.py" "$TMP_STORAGE/mettagrid/"
  cp "$(git rev-parse --show-toplevel)/Makefile" "$TMP_STORAGE/"
  cp "$(git rev-parse --show-toplevel)/pyproject.toml" "$TMP_STORAGE/"
  cp "$(git rev-parse --show-toplevel)/requirements_pinned.txt" "$TMP_STORAGE/"
  cp "$(git rev-parse --show-toplevel)/requirements.txt" "$TMP_STORAGE/"
  cp "$(git rev-parse --show-toplevel)/setup.py" "$TMP_STORAGE/"
  
  # Copy git utilities module
  echo "Copying git utilities module..."
  cp "$(git rev-parse --show-toplevel)/metta/util/git.py" "$TMP_STORAGE/metta/util/"
  cp "$(git rev-parse --show-toplevel)/metta/util/__init__.py" "$TMP_STORAGE/metta/util/" 2>/dev/null || touch "$TMP_STORAGE/metta/util/__init__.py"
  
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
echo "============================================================" >> "$LOG_FILE"

# Save the current branch so we can return to it at the end
ORIGINAL_BRANCH=$(git symbolic-ref --quiet HEAD || git rev-parse HEAD)
if [[ $ORIGINAL_BRANCH == refs/heads/* ]]; then
  ORIGINAL_BRANCH=${ORIGINAL_BRANCH#refs/heads/}
fi
echo "Original branch: $ORIGINAL_BRANCH" >> "$LOG_FILE"

# Get list of commits to process - use rev-list to avoid duplicates
COMMITS=$(git rev-list --max-count=$NUM_COMMITS HEAD)

# Count actual commits found
COMMIT_COUNT=$(echo "$COMMITS" | wc -l | tr -d ' ')
echo "Found $COMMIT_COUNT commits to process"

# Start processing commits
echo "Starting to process $COMMIT_COUNT commits..."
COUNTER=1
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
  cp -rf "$TMP_STORAGE/devops"/* "$REPO_ROOT/devops/" 2>/dev/null || :
  
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
  cp -f "$TMP_STORAGE/requirements_pinned.txt" "$REPO_ROOT/"
  cp -f "$TMP_STORAGE/requirements.txt" "$REPO_ROOT/"
  cp -f "$TMP_STORAGE/setup.py" "$REPO_ROOT/"
  
  # Check if experience.py exists and patch it
  EXPERIENCE_PATH="$REPO_ROOT/metta/rl/pufferlib/experience.py"
  if [ -f "$EXPERIENCE_PATH" ]; then
    echo "Patching metta/rl/pufferlib/experience.py"
    
    # Find the flatten_batch function and replace it
    awk -v RS='def flatten_batch' -v ORS='def flatten_batch' 'NR==1{print $0} NR==2{print "\n"; system("cat '$TMP_STORAGE'/experience_patch.py"); next} NR>2{exit}' "$EXPERIENCE_PATH" > "$EXPERIENCE_PATH.new"
    
    # If the new file is created and has content, replace the old one
    if [ -s "$EXPERIENCE_PATH.new" ]; then
      mv "$EXPERIENCE_PATH.new" "$EXPERIENCE_PATH"
      echo "Successfully patched experience.py"
    else
      echo "Warning: Failed to patch experience.py - file may have unexpected format"
      # If the replacement didn't work, let's try a different approach - add the function at the end
      echo "Attempting alternative patching method..."
      cat "$TMP_STORAGE/experience_patch.py" >> "$EXPERIENCE_PATH"
      echo "Added the function to the end of experience.py"
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
    git add -f "$REPO_ROOT/requirements_pinned.txt"
    git add -f "$REPO_ROOT/requirements.txt"
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
      trainer.env=env/mettagrid/simple
    
    # Log job details
    echo "Job submitted for commit: $COMMIT_SHORT" | tee -a "$LOG_FILE"
    echo "  Branch: $TEST_BRANCH" | tee -a "$LOG_FILE"
    echo "  Run ID: $RUN_NAME" | tee -a "$LOG_FILE"
    
  else
    echo "Warning: experience.py doesn't exist in commit $COMMIT_SHORT. Skipping..."
    echo "experience.py NOT FOUND in commit $COMMIT_SHORT. Skipped." >> "$LOG_FILE"
  fi
  
  COUNTER=$((COUNTER + 1))
  
  # Clean up any Python cache files or other untracked files before moving to the next commit
  echo "Cleaning up before moving to the next commit..."
  find "$REPO_ROOT" -name "__pycache__" -type d -exec rm -rf {} +  2>/dev/null || :
  find "$REPO_ROOT" -name "*.pyc" -delete 2>/dev/null || :
  find "$REPO_ROOT" -name "*.pyo" -delete 2>/dev/null || :
  find "$REPO_ROOT" -name ".DS_Store" -delete 2>/dev/null || :
  
  # Force clean untracked files
  git clean -fd
done

# Return to the original branch
echo "Returning to original branch: $ORIGINAL_BRANCH"
git checkout "$ORIGINAL_BRANCH"

echo ""
echo "============================================================"
echo "Debug job submission complete!"
echo "Processed $COMMIT_COUNT commits."
echo "Log file: $LOG_FILE"
echo "============================================================"
echo ""
echo "Review the log file to see all the jobs that were submitted."
exit 0