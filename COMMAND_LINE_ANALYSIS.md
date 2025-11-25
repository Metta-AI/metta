# Command Line Analysis Guide

Quick reference for investigating and fixing the LP performance degradation from the command line.

## TL;DR - Quick Fix

```bash
# 1. Apply the fix (automatic)
./tools/fix_lp_performance.sh fix

# 2. Test it
timeout 180s uv run ./tools/run.py experiment.cogs_v_clips use_lp=True run=test_fix

# 3. Check SPS in logs
grep -i sps outputs/*test_fix*/logs/train.log | tail -20
```

## Full Investigation Workflow

### Step 1: Confirm the Problem (Optional)

Run baseline tests to see the issue:

```bash
# Test DiscreteRandom (should be fast)
timeout 180s uv run ./tools/run.py experiment.cogs_v_clips \
    use_lp=False \
    run=baseline_discrete

# Test Learning Progress (should be slow and degrade)
timeout 180s uv run ./tools/run.py experiment.cogs_v_clips \
    use_lp=True \
    run=baseline_lp

# Compare SPS
echo "DiscreteRandom SPS:"
grep -iE "sps:|steps_per_second" outputs/*baseline_discrete*/logs/train.log | head -10

echo "Learning Progress SPS:"
grep -iE "sps:|steps_per_second" outputs/*baseline_lp*/logs/train.log | head -10
```

### Step 2: Apply the Fix

**Option A: Automated (recommended)**

```bash
./tools/fix_lp_performance.sh fix
```

**Option B: Manual**

Edit `metta/cogworks/curriculum/learning_progress_algorithm.py` around line 490:

```python
def update_task_performance(self, task_id: int, score: float) -> None:
    """Update task performance atomically."""
    self.task_tracker.update_task_performance_with_bidirectional_emas(
        task_id=task_id,
        score=score,
        scorer=self.scorer if hasattr(self.scorer, "config") else None,
    )

    # NEW: Batch invalidations to reduce get_all_tracked_tasks() calls
    if not hasattr(self, '_updates_since_invalidation'):
        self._updates_since_invalidation = 0

    self._updates_since_invalidation += 1

    if self._updates_since_invalidation >= 100:
        self.scorer.invalidate_cache()
        self._updates_since_invalidation = 0

    # OLD (comment out): self.scorer.invalidate_cache()

    self.invalidate_cache()
```

### Step 3: Test the Fix

```bash
timeout 180s uv run ./tools/run.py experiment.cogs_v_clips \
    use_lp=True \
    run=fixed_lp

# Check results
grep -iE "sps:|steps_per_second" outputs/*fixed_lp*/logs/train.log | head -20
```

### Step 4: Compare Results

```bash
./tools/fix_lp_performance.sh compare
```

Or manually:

```bash
echo "=== SPS Comparison ==="
echo ""
echo "DiscreteRandom (baseline):"
grep -iE "sps:|steps_per_second" outputs/*baseline_discrete*/logs/train.log | \
    awk '{print $NF}' | head -10

echo ""
echo "LP Before Fix:"
grep -iE "sps:|steps_per_second" outputs/*baseline_lp*/logs/train.log | \
    awk '{print $NF}' | head -10

echo ""
echo "LP After Fix:"
grep -iE "sps:|steps_per_second" outputs/*fixed_lp*/logs/train.log | \
    awk '{print $NF}' | head -10
```

## Using the Helper Script

The `fix_lp_performance.sh` script automates the workflow:

```bash
# Full workflow
./tools/fix_lp_performance.sh baseline   # Run baseline tests
./tools/fix_lp_performance.sh fix        # Apply fix
./tools/fix_lp_performance.sh test       # Test fix
./tools/fix_lp_performance.sh compare    # Compare all results

# Quick workflow (shorter tests)
./tools/fix_lp_performance.sh quick      # Quick baseline
./tools/fix_lp_performance.sh fix        # Apply fix
./tools/fix_lp_performance.sh test       # Test fix

# Other commands
./tools/fix_lp_performance.sh revert     # Undo the fix
./tools/fix_lp_performance.sh help       # Show help
```

## Alternative: Test Different Approaches

### Test with Smaller Task Pool

```bash
timeout 180s uv run ./tools/run.py experiment.cogs_v_clips \
    use_lp=True \
    num_active_tasks=100 \
    run=small_pool_test
```

### Test with Different Batch Sizes

After applying the fix, you can tune the batch size by editing line 490:

```python
if self._updates_since_invalidation >= 50:  # Try 50, 100, 200
    self.scorer.invalidate_cache()
    self._updates_since_invalidation = 0
```

## Profiling (Advanced)

### Add Simple Profiling

Add to `metta/cogworks/curriculum/lp_scorers.py` in `BidirectionalLPScorer.__init__`:

```python
def __init__(self, config, tracker):
    super().__init__(config, tracker)
    self._prof_calls = 0
    import time
    self._prof_last_log = time.time()
```

And in `_calculate_task_distribution`:

```python
def _calculate_task_distribution(self, tracker):
    self._prof_calls += 1
    import time
    import logging
    logger = logging.getLogger(__name__)

    if time.time() - self._prof_last_log > 10.0:
        logger.warning(f"[LP_PROF] _calculate_task_distribution called {self._prof_calls} times")
        self._prof_last_log = time.time()

    # ... rest of method
```

Then check logs:

```bash
grep "LP_PROF" outputs/*/logs/train.log
```

### Profile with py-spy

```bash
# Find the training process PID
ps aux | grep "tools/run.py"

# Profile for 60 seconds
py-spy record -o profile.svg --pid <PID>

# View profile.svg in browser
open profile.svg
```

## Expected Results

**Before Fix:**

- Initial SPS: ~15,000
- After 50 epochs: ~8,000
- After 100 epochs: ~5,000
- Clear degradation trend

**After Fix:**

- Initial SPS: ~15,000
- After 50 epochs: ~14,000
- After 100 epochs: ~13,000
- Minimal degradation (similar to DiscreteRandom)

## Troubleshooting

### Fix doesn't help

If the fix doesn't improve performance:

1. Check the fix was applied correctly:

   ```bash
   grep -A 10 "updates_since_invalidation" \
       metta/cogworks/curriculum/learning_progress_algorithm.py
   ```

2. Try a smaller batch size (50 instead of 100)

3. Add profiling to find other bottlenecks

### Can't find SPS in logs

Try different patterns:

```bash
# Try various SPS patterns
grep -i "sps" outputs/*/logs/train.log
grep "steps" outputs/*/logs/train.log
grep "per.*second" outputs/*/logs/train.log

# Or check WandB logs if configured
```

### Tests timing out

Reduce timeout or run for fewer epochs:

```bash
timeout 90s uv run ./tools/run.py experiment.cogs_v_clips \
    use_lp=True \
    trainer.total_timesteps=1000000 \
    run=quick_test
```

## Reverting Changes

```bash
# If using the script
./tools/fix_lp_performance.sh revert

# Or manually with git
git checkout metta/cogworks/curriculum/learning_progress_algorithm.py
```

## Understanding the Fix

**Root Cause**: `get_all_tracked_tasks()` scans all shared memory slots on every episode completion.

**The Fix**: Batch cache invalidations - only scan every 100 episodes instead of every episode.

**Impact**:

- Before: ~5000 memory scans/epoch (one per episode)
- After: ~50 memory scans/epoch (once per 100 episodes)
- **100x reduction** in expensive operations

**Trade-off**: LP scores can be up to 100 episodes stale, but this is acceptable because:

- The LP signal is noisy anyway
- 100 episodes is a small fraction of training
- Learning progress changes gradually

## Next Steps After Fix

1. **Verify training quality**: Check that learning curves are still good
2. **Tune batch size**: Try 50, 100, 200 to find optimal trade-off
3. **Add to production configs**: Update all LP configs with the fix
4. **Consider caching**: Add `get_all_tracked_tasks()` caching as complementary fix

## Files Modified

- `metta/cogworks/curriculum/learning_progress_algorithm.py` - Apply batch invalidation
- Optional: `metta/cogworks/curriculum/task_tracker.py` - Add caching

## Related Documentation

- `performance_degradation_investigation.md` - Full investigation details
- `NEXT_STEPS_PERF_DEBUG.md` - Detailed debugging guide
- `tools/debug_performance.py` - Automated performance testing
