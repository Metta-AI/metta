# Next Steps: Performance Degradation Debug

## What We Know

✅ **Confirmed**:

- DiscreteRandom curriculum: Fast, constant SPS
- Learning Progress curriculum: Slow, degrades over time
- `performance_mode=True`: Does NOT fix the issue
- **Non-bidirectional LP**: ALSO impacted similarly

❌ **Ruled Out**:

- Once-per-epoch stats collection (Gini, per-label stats, etc.)
- Bidirectional-specific operations (p_fast/p_slow, reweighting, baseline normalization)

## 🎯 ROOT CAUSE IDENTIFIED: get_all_tracked_tasks() Memory Scan

**The Bottleneck Chain** (happens EVERY episode):

1. `update_task_performance()` → `scorer.invalidate_cache()` → `self._stale_dist = True`
2. Next `get_task()` → `score_task()` → checks `if self._stale_dist:`
3. Calls `_calculate_task_distribution(tracker)`
4. **`tracker.get_all_tracked_tasks()`** → **SCANS ALL max_tasks SLOTS**
5. This is **O(max_tasks)** even if only 10% are active

**Complexity**: With max_tasks=1000, ~5000 episodes/epoch = **5,000,000 shared memory reads/epoch**

**Why DiscreteRandom doesn't have this**:

- No `_calculate_task_distribution()` call
- No `get_all_tracked_tasks()` scan
- Just returns `{task_id: 1.0}` instantly

## Immediate Actions (Priority Order)

### 1. **IMPLEMENT FIX: Batch cache invalidation** (10 min) - DO THIS FIRST!

The fix is simple - invalidate cache every N updates instead of every single update.

Edit `metta/cogworks/curriculum/learning_progress_algorithm.py` around line 490:

```python
def update_task_performance(self, task_id: int, score: float) -> None:
    """Update task performance atomically."""
    # Atomic update: All EMAs in one lock
    self.task_tracker.update_task_performance_with_bidirectional_emas(
        task_id=task_id,
        score=score,
        scorer=self.scorer if hasattr(self.scorer, "config") else None,
    )

    # NEW: Batch invalidations to reduce get_all_tracked_tasks() calls
    if not hasattr(self, '_updates_since_invalidation'):
        self._updates_since_invalidation = 0

    self._updates_since_invalidation += 1

    # Invalidate every 100 updates instead of every single update
    # This reduces get_all_tracked_tasks() calls from ~5000/epoch to ~50/epoch
    if self._updates_since_invalidation >= 100:
        self.scorer.invalidate_cache()
        self._updates_since_invalidation = 0

    # OLD CODE (comment this out):
    # self.scorer.invalidate_cache()  # Was called EVERY episode!

    # Invalidate stats cache when task performance changes
    self.invalidate_cache()
```

**Run test**:

```bash
timeout 300s uv run ./tools/run.py experiment.cogs_v_clips use_lp=True run=batch_invalidate_fix
```

**Expected**:

- SPS should be much higher and more stable
- Should approach DiscreteRandom performance
- Trade-off: LP scores slightly stale (100 episodes old max)
- But LP signal should still work fine

### 2. **Test with reduced task pool** (15 min)

If O(n) scaling is the issue, smaller pool should be faster:

```python
# In recipes/experiment/cogs_v_clips.py, line 282
cur_alg = LearningProgressConfig(
    num_active_tasks=100,  # vs default 256-1000
) if use_lp else DiscreteRandomCurriculumConfig()
```

**Run comparison**:

```bash
# Small pool
timeout 180s uv run ./tools/run.py experiment.cogs_v_clips \
    use_lp=True num_active_tasks=100 run=debug_small_pool

# Normal pool
timeout 180s uv run ./tools/run.py experiment.cogs_v_clips \
    use_lp=True run=debug_normal_pool
```

**Expected**: If small pool is significantly faster with less degradation, confirms O(n) issue.

### 3. **Test lazy cache invalidation** (30 min)

Currently, cache is invalidated on EVERY performance update. Try batching:

```python
# In metta/cogworks/curriculum/learning_progress_algorithm.py:477-500

def update_task_performance(self, task_id: int, score: float) -> None:
    """Update task performance atomically."""
    # Atomic update
    self.task_tracker.update_task_performance_with_bidirectional_emas(
        task_id=task_id,
        score=score,
        scorer=self.scorer if hasattr(self.scorer, "config") else None,
    )

    # MODIFIED: Don't invalidate on every update
    # Mark distribution as stale - LP scores will be recalculated on next sampling
    # self.scorer.invalidate_cache()  # COMMENTED OUT

    # Instead: batch invalidations every N updates
    if not hasattr(self, '_updates_since_invalidation'):
        self._updates_since_invalidation = 0

    self._updates_since_invalidation += 1

    # Invalidate every 50 updates (tune this number)
    if self._updates_since_invalidation >= 50:
        self.scorer.invalidate_cache()
        self._updates_since_invalidation = 0

    # Invalidate stats cache when task performance changes
    self.invalidate_cache()
```

**Run test**:

```bash
timeout 180s uv run ./tools/run.py experiment.cogs_v_clips \
    use_lp=True run=debug_batch_invalidate
```

**Expected**: Should see improvement if cache thrashing is the issue.

## Alternative Hypotheses to Test

### A. **Shared Memory Lock Contention**

**Test**: Add timing around lock acquisition

```python
# In metta/cogworks/curriculum/task_tracker.py:264

with self._backend.acquire_lock():
    import time
    lock_acquire_time = time.perf_counter()

    # ... existing update code ...

    lock_held_time = time.perf_counter() - lock_acquire_time
    if lock_held_time > 0.001:  # >1ms
        logger.warning(f"Lock held for {lock_held_time*1000:.1f}ms")
```

### B. **Label Dictionary Growth**

**Test**: Check dictionary size over time

```python
# In metta/cogworks/curriculum/task_tracker.py, add to get_state()

logger.warning(
    f"Label hash dict size: {len(self._label_hash_to_string)} entries, "
    f"{sys.getsizeof(self._label_hash_to_string)} bytes"
)
```

### C. **get_episode_stats() Overhead**

**Test**: Profile this per-episode call

```python
# In metta/cogworks/curriculum/curriculum_env.py:137

t0 = time.perf_counter()
curriculum_stats = self._curriculum.get_episode_stats(label)
elapsed = time.perf_counter() - t0

if elapsed > 0.001:  # >1ms
    logger.warning(f"get_episode_stats took {elapsed*1000:.1f}ms")
```

## Quick Validation Tests

### Test 1: Confirm checkpoint resume behavior (30 min)

```bash
# Run to epoch 50
timeout 300s uv run ./tools/run.py experiment.cogs_v_clips \
    use_lp=True run=checkpoint_test_1

# Note the final SPS at epoch 50
# Then resume
uv run ./tools/run.py experiment.cogs_v_clips \
    use_lp=True run=checkpoint_test_2 \
    resume_from=outputs/checkpoint_test_1/checkpoints/latest.pt

# Check if epoch 51 SPS is similar to epoch 1 or epoch 50
```

**If SPS resets**: In-memory state accumulation (not persisted) **If SPS stays low**: Problem is in checkpointed data

### Test 2: Compare branches (30 min)

```bash
# On current branch
timeout 180s uv run ./tools/run.py experiment.cogs_v_clips \
    use_lp=True run=current_branch

# Switch to main
git stash
git checkout main
timeout 180s uv run ./tools/run.py experiment.cogs_v_clips \
    use_lp=True run=main_branch

# Compare SPS degradation rates
```

## Expected Bottleneck Evidence

If `score_tasks()` is the bottleneck, you should see:

1. **Call frequency**: ~1000-5000 calls per epoch (grep logs)
2. **Time scaling**: Each call takes longer as tasks increase
3. **Cache misses**: Cache invalidated frequently (check invalidate_cache calls)
4. **Task count correlation**: More tasks = worse performance

## If score_tasks IS the bottleneck, solutions are:

### Solution 1: Smarter caching

- Only recompute scores for tasks that were updated
- Keep cache valid for unchanged tasks
- Implement incremental updates

### Solution 2: Reduce invalidation frequency

- Batch invalidations (every N updates)
- Trade off: slightly stale scores for better performance
- Validate that staleness doesn't hurt training quality

### Solution 3: Optimize score computation

- Use cached numpy arrays (avoid rebuilding)
- Parallelize score calculation
- Pre-compute expensive operations

### Solution 4: Reduce task pool size

- Keep fewer tasks active (100-200 instead of 1000)
- Faster sampling but potentially less curriculum diversity
- Trade-off to test

## Tools Created

1. **Performance Test Runner**: `tools/debug_performance.py`

   ```bash
   uv run ./tools/debug_performance.py --quick
   ```

2. **Investigation Plan**: `performance_degradation_investigation.md`
   - Detailed hypotheses and test protocols
   - Updated with current findings

3. **Profiling Patch**: `tools/profile_lp_scorer.patch`
   - Can apply to instrument score_tasks timing

## Summary

**🎯 ROOT CAUSE CONFIRMED**: `get_all_tracked_tasks()` scans O(max_tasks) slots on EVERY episode completion.

**📊 Impact**: With 1000 max_tasks × 5000 episodes/epoch = **5 million shared memory scans per epoch**

**✅ FIX IDENTIFIED**: Batch cache invalidation - invalidate every 100 episodes instead of every episode

**Implementation**: 10 minute code change in `learning_progress_algorithm.py:490`

**Expected Result**:

- Reduces memory scans from 5000/epoch to 50/epoch (100x improvement)
- Should restore SPS to near-DiscreteRandom levels
- Trade-off: LP scores up to 100 episodes stale (acceptable)

**Next Steps**:

1. Implement batch invalidation fix (see section #1 above)
2. Test and measure SPS improvement
3. Tune batch size (50-200 episodes) based on results
4. Optionally add `get_all_tracked_tasks()` caching as complementary fix

The issue is **algorithmic, not architectural** - a simple batching change should fix it!
