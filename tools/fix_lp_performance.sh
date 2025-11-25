#!/bin/bash
# Quick fix for Learning Progress performance degradation
# Root cause: get_all_tracked_tasks() called on every episode completion

set -e

echo "=================================="
echo "LP Performance Fix Tool"
echo "=================================="
echo ""
echo "This script helps you:"
echo "1. Run baseline tests (before fix)"
echo "2. Apply the batch invalidation fix"
echo "3. Test the fix"
echo "4. Compare results"
echo ""

# Function to run a test
run_test() {
  local name=$1
  local timeout=$2
  local extra_args=$3

  echo ""
  echo "Running: $name"
  echo "Command: timeout ${timeout}s uv run ./tools/run.py cogs_v_clips.train run=$name $extra_args"
  echo "Press Ctrl+C within 5 seconds to skip..."
  sleep 5

  # Execute the command - split extra_args properly
  timeout ${timeout}s uv run ./tools/run.py cogs_v_clips.train run=$name ${extra_args} || true

  echo ""
  echo "Test complete: $name"
  echo "Check logs in: outputs/"
  echo ""
}

# Function to extract SPS from logs
show_sps() {
  local run_name=$1
  echo ""
  echo "SPS data for $run_name:"
  echo "=================================="

  # Find the most recent matching log
  local log_file=$(ls -t outputs/*${run_name}*/logs/train.log 2> /dev/null | head -1)

  if [ -z "$log_file" ]; then
    echo "No logs found for $run_name"
    return
  fi

  # Extract SPS values (looking for patterns like "SPS: 1234" or "steps_per_second: 1234")
  echo "Extracting SPS samples..."
  grep -iE "(sps:|steps_per_second)" "$log_file" | head -20 | tail -10

  echo ""
  echo "Log file: $log_file"
  echo ""
}

# Parse command line arguments
ACTION=${1:-help}

case "$ACTION" in
  "baseline")
    echo "=== BASELINE TESTS ==="
    echo "Running baseline comparisons to confirm the issue"
    echo ""

    echo "Test 1: DiscreteRandom (should be fast, constant SPS)"
    run_test "msb_perfdiagnosis_baseline_discrete" 600 "use_lp=False"
    show_sps "msb_perfdiagnosis_baseline_discrete"

    echo "Test 2: Learning Progress (should be slow, degrading SPS)"
    run_test "msb_perfdiagnosis_baseline_lp" 600 "use_lp=True"
    show_sps "msb_perfdiagnosis_baseline_lp"

    echo ""
    echo "=== BASELINE COMPLETE ==="
    echo "Compare the SPS trends above."
    echo "If LP is significantly slower and degrades, proceed to 'fix' step."
    ;;

  "fix")
    echo "=== APPLYING FIX ==="
    echo ""

    # Backup the file
    FILE="metta/cogworks/curriculum/learning_progress_algorithm.py"
    BACKUP="${FILE}.backup_$(date +%Y%m%d_%H%M%S)"

    echo "Creating backup: $BACKUP"
    cp "$FILE" "$BACKUP"

    echo ""
    echo "Applying batch invalidation fix..."
    echo ""

    # Check if fix is already applied
    if grep -q "_updates_since_invalidation" "$FILE"; then
      echo "✓ Fix appears to be already applied!"
      echo "  Found '_updates_since_invalidation' in the file"
      echo ""
      echo "=== FIX APPLIED ==="
      echo "Backup saved: $BACKUP"
      echo "Next: Run './tools/fix_lp_performance.sh test' to test the fix"
      return 0
    fi

    # Apply the fix using a simple sed replacement
    echo "Replacing 'self.scorer.invalidate_cache()' with batched version..."

    # Create a Python script to apply the fix more safely
    cat > /tmp/apply_lp_fix.py << 'PYTHON_SCRIPT'
import sys

def apply_fix(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Find the line with self.scorer.invalidate_cache() in update_task_performance
    new_lines = []
    i = 0
    found = False

    while i < len(lines):
        line = lines[i]

        # Look for the scorer.invalidate_cache() call after the EMA update
        if ('self.scorer.invalidate_cache()' in line and
            i > 0 and 'update_task_performance_with_bidirectional_emas' in ''.join(lines[max(0, i-10):i])):

            found = True
            indent = len(line) - len(line.lstrip())
            spaces = ' ' * indent

            # Add the batched invalidation code
            new_lines.append(f'{spaces}# PERFORMANCE FIX: Batch invalidations to reduce get_all_tracked_tasks() calls\n')
            new_lines.append(f'{spaces}# Only invalidate cache every N updates instead of every single update\n')
            new_lines.append(f"{spaces}if not hasattr(self, '_updates_since_invalidation'):\n")
            new_lines.append(f'{spaces}    self._updates_since_invalidation = 0\n')
            new_lines.append(f'{spaces}\n')
            new_lines.append(f'{spaces}self._updates_since_invalidation += 1\n')
            new_lines.append(f'{spaces}\n')
            new_lines.append(f'{spaces}# Invalidate every 100 updates (reduces scans from ~5000/epoch to ~50/epoch)\n')
            new_lines.append(f'{spaces}if self._updates_since_invalidation >= 100:\n')
            new_lines.append(f'{spaces}    {line.strip()}\n')
            new_lines.append(f'{spaces}    self._updates_since_invalidation = 0\n')
            new_lines.append(f'{spaces}\n')
            new_lines.append(f'{spaces}# OLD: {line.strip()}  # Was called EVERY episode\n')
        else:
            new_lines.append(line)

        i += 1

    if not found:
        print("ERROR: Could not find 'self.scorer.invalidate_cache()' to replace!")
        print("The code may have changed. Apply fix manually.")
        sys.exit(1)

    with open(filename, 'w') as f:
        f.writelines(new_lines)

    print("✓ Fix applied successfully!")
    return True

if __name__ == '__main__':
    apply_fix(sys.argv[1])
PYTHON_SCRIPT

    python3 /tmp/apply_lp_fix.py "$FILE" || {
      echo ""
      echo "⚠️  Automatic fix failed. Apply manually:"
      echo ""
      echo "Edit: $FILE"
      echo "Around line 490, find:"
      echo "    self.scorer.invalidate_cache()"
      echo ""
      echo "Replace with:"
      echo "    # Batch invalidations"
      echo "    if not hasattr(self, '_updates_since_invalidation'):"
      echo "        self._updates_since_invalidation = 0"
      echo "    self._updates_since_invalidation += 1"
      echo "    if self._updates_since_invalidation >= 100:"
      echo "        self.scorer.invalidate_cache()"
      echo "        self._updates_since_invalidation = 0"
      echo ""
      exit 1
    }

    echo "=== FIX APPLIED ==="
    echo "Backup saved: $BACKUP"
    echo "Next: Run './tools/fix_lp_performance.sh test' to test the fix"
    ;;

  "test")
    echo "=== TESTING FIX ==="
    echo "Running with the fix applied"
    echo ""

    run_test "msb_perfdiagnosis_fixed_lp" 600 "use_lp=True"
    show_sps "msb_perfdiagnosis_fixed_lp"

    echo ""
    echo "=== TEST COMPLETE ==="
    echo "Compare SPS with baseline_lp results."
    echo "Expected: Much higher SPS, less degradation"
    ;;

  "compare")
    echo "=== COMPARING RESULTS ==="
    echo ""

    echo "DiscreteRandom (baseline):"
    show_sps "msb_perfdiagnosis_baseline_discrete"

    echo ""
    echo "LP Before Fix:"
    show_sps "msb_perfdiagnosis_baseline_lp"

    echo ""
    echo "LP After Fix:"
    show_sps "msb_perfdiagnosis_fixed_lp"

    echo ""
    echo "=== COMPARISON TIPS ==="
    echo "1. Compare initial SPS across all three"
    echo "2. Check if SPS degrades over time (first few epochs vs last few)"
    echo "3. Fixed LP should be similar to DiscreteRandom"
    ;;

  "revert")
    echo "=== REVERTING FIX ==="

    FILE="metta/cogworks/curriculum/learning_progress_algorithm.py"
    BACKUP=$(ls -t ${FILE}.backup_* 2> /dev/null | head -1)

    if [ -z "$BACKUP" ]; then
      echo "No backup found!"
      echo "You may need to git checkout the file:"
      echo "  git checkout $FILE"
      exit 1
    fi

    echo "Restoring from: $BACKUP"
    cp "$BACKUP" "$FILE"
    echo "✓ Reverted successfully"
    ;;

  "quick")
    echo "=== QUICK TEST ==="
    echo "Running short tests to quickly verify the fix"
    echo ""

    echo "1. Quick baseline (no fix)"
    run_test "msb_perfdiagnosis_quick_baseline" 300 "use_lp=True"

    echo ""
    echo "Now apply the fix with: ./tools/fix_lp_performance.sh fix"
    echo "Then run: ./tools/fix_lp_performance.sh test"
    ;;

  "help" | *)
    echo "Usage: ./tools/fix_lp_performance.sh [command]"
    echo ""
    echo "Commands:"
    echo "  baseline   - Run baseline tests (DiscreteRandom + LP before fix)"
    echo "  fix        - Apply the batch invalidation fix"
    echo "  test       - Test the fix (run LP with fix applied)"
    echo "  compare    - Show SPS comparison across all tests"
    echo "  revert     - Revert the fix (restore from backup)"
    echo "  quick      - Quick test (shorter timeout)"
    echo "  help       - Show this help"
    echo ""
    echo "Typical workflow:"
    echo "  1. ./tools/fix_lp_performance.sh baseline"
    echo "  2. ./tools/fix_lp_performance.sh fix"
    echo "  3. ./tools/fix_lp_performance.sh test"
    echo "  4. ./tools/fix_lp_performance.sh compare"
    echo ""
    echo "Quick workflow:"
    echo "  1. ./tools/fix_lp_performance.sh quick"
    echo "  2. ./tools/fix_lp_performance.sh fix"
    echo "  3. ./tools/fix_lp_performance.sh test"
    echo ""
    exit 0
    ;;
esac

echo ""
echo "Done!"
