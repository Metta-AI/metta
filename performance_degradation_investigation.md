# Performance Degradation Investigation Plan

## Problem Summary

- **Issue**: Using `use_LP=True` (Learning Progress curriculum) causes major performance hit that worsens over time
  (measured in steps/second)
- **Context**:
  - Lesser hit on main branch with same code
  - Almost no hit with DiscreteRandomCurriculum
  - Reportedly not affected by checkpoint resume (needs confirmation)
  - Branch: `msb_curr_min_v1`

## Hypothesis: Primary Suspects

### ⚠️ CRITICAL UPDATES:

1. **`performance_mode=True` does NOT fix the issue!**
   - Rules out: Once-per-epoch stats collection (Gini, per-label stats, etc.)

2. **Non-bidirectional LP is ALSO impacted similarly!**
   - Rules out: Bidirectional-specific operations (p_fast/p_slow, reweighting, baseline normalization)

**Conclusion**: The bottleneck is in **CORE CURRICULUM MECHANICS** shared by both LP implementations.

### 1. **~~Stats Collection Overhead~~** (RULED OUT)

**User tested**: `performance_mode=True` doesn't significantly help

### 2. **~~Bidirectional-Specific Operations~~** (RULED OUT)

**User tested**: Non-bidirectional LP also degrades

**What this rules out**:

- Bidirectional EMA calculations (p_fast, p_slow)
- Baseline normalization
- Reweighting functions
- Early progress amplification

**What remains** (operations common to BOTH LP variants):

- `score_tasks()` frequency and overhead
- Cache invalidation pattern
- EMA updates in TaskTracker
- Shared memory lock contention
- O(n) operations over growing task pool

### 3. **🎯 SMOKING GUN: get_all_tracked_tasks() Called on Every Episode**

**✓ Affects both bidirectional AND basic LP** (user confirmed)

**The Pattern** (every episode completion):

1. `update_task_performance()` → `scorer.invalidate_cache()` → `self._stale_dist = True`
2. Next `get_task()` → `_choose_task()` → `score_task()`
3. `score_task()` checks `if self._stale_dist:` → calls `_calculate_task_distribution()`
4. **`_calculate_task_distribution()` → `tracker.get_all_tracked_tasks()`**
5. **`get_all_tracked_tasks()` → SCANS ENTIRE SHARED MEMORY ARRAY O(max_tasks)**

**The Bottleneck Code**:

```python
# metta/cogworks/curriculum/lp_scorers.py:407-413
def _calculate_task_distribution(self, tracker: TaskTracker):
    task_ids = tracker.get_all_tracked_tasks()  # ← EXPENSIVE!
    # ... then process all tasks with numpy operations
    # ... then write back LP scores to shared memory

# metta/cogworks/curriculum/task_tracker.py:490-510
def get_all_tracked_tasks(self) -> List[int]:
    if isinstance(self._backend, SharedMemoryBackend):
        task_ids = []
        for i in range(self._backend.max_tasks):  # ← O(max_tasks) scan!
            state = self._backend.get_task_state(i)  # Shared memory read
            if bool(state.is_active):
                task_ids.append(int(state.task_id))
        return task_ids
```

**Complexity Analysis**:

- **Per episode**: O(max_tasks) memory scan + O(active_tasks) numpy ops + O(active_tasks) writes
- **Per epoch**: ~5000 episodes × O(max_tasks) = **5M+ shared memory operations**
- **With max_tasks=1000**: 5,000,000 shared memory reads per epoch!
- **Gets worse over time**: More episodes/epoch as agent gets better

**Why DiscreteRandom doesn't have this**:

```python
# DiscreteRandom.score_tasks() - no distribution calculation
def score_tasks(self, task_ids):
    return {task_id: 1.0 for task_id in task_ids}  # O(n), no memory scan
```

**Files**:

- `metta/cogworks/curriculum/lp_scorers.py:407-471` - `_calculate_task_distribution()` (THE BOTTLENECK)
- `metta/cogworks/curriculum/task_tracker.py:490-510` - `get_all_tracked_tasks()` (EXPENSIVE SCAN)
- `metta/cogworks/curriculum/learning_progress_algorithm.py:477-500` - Invalidation trigger

### 3. **Shared Memory Lock Contention**

**Evidence**:

- Every episode completion acquires lock for EMA updates
- Lock held during multi-field atomic updates
- More tasks = more updates = more lock contention
- **Gets worse over time** as completion rate increases

**Files**:

- `metta/cogworks/curriculum/task_tracker.py:238-360` - Lock acquisition in hot path
- `metta/cogworks/curriculum/shared_memory_backend.py:270-395` - SharedMemoryBackend locking

**Why DiscreteRandom doesn't have this**:

- DiscreteRandom doesn't track task performance
- No EMA updates
- Minimal locking

### 4. **Label Hash Dictionary Growth**

**Evidence**:

- `_label_hash_to_string` dictionary grows unbounded
- Every new task adds to this dictionary
- Dictionary lookups get slower with size
- Never cleaned up (even when tasks evicted)

**Files**:

- `metta/cogworks/curriculum/task_tracker.py:86` - `_label_hash_to_string` initialization
- `metta/cogworks/curriculum/task_tracker.py:553-579` - `set_task_label()` adds entries
- `metta/cogworks/curriculum/task_tracker.py:580-599` - `get_task_label()` lookup

### 5. **get_episode_stats() on Every Completion**

**Evidence**:

- Called on EVERY episode completion
- Gets eviction counts (may scan dictionaries)
- Called from curriculum_env hot path

**Files**:

- `metta/cogworks/curriculum/curriculum_env.py:137` - Called per episode
- `metta/cogworks/curriculum/curriculum.py:331-354` - Implementation

## Investigation Protocol (UPDATED - Performance Mode Doesn't Help)

### Phase 1: ✅ COMPLETED - Problem Confirmed

- ✅ Discrete curriculum is fast
- ✅ LP curriculum degrades over time
- ✅ `performance_mode=True` does NOT fix it
- **Conclusion**: Issue is in HOT PATH (per-episode operations)

### Phase 2: Profile LP Score Recalculation (30 min) - **DO THIS FIRST**

This is now the prime suspect. Profile how often and how long `score_tasks()` takes:

```python
# Add to metta/cogworks/curriculum/lp_scorers.py:BidirectionalLPScorer
import time

class BidirectionalLPScorer(LPScorer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._score_call_count = 0
        self._total_score_time = 0.0
        self._last_log_time = time.time()

    def score_tasks(self, task_ids: list[int]) -> dict[int, float]:
        self._score_call_count += 1
        t0 = time.perf_counter()

        result = super().score_tasks(task_ids)  # Or existing implementation

        elapsed = time.perf_counter() - t0
        self._total_score_time += elapsed

        # Log every 10 seconds
        if time.time() - self._last_log_time > 10.0:
            avg_time = self._total_score_time / max(1, self._score_call_count)
            logger.warning(
                f"LP Scoring: {self._score_call_count} calls, "
                f"avg={avg_time*1000:.2f}ms, "
                f"total={self._total_score_time:.2f}s, "
                f"tasks={len(task_ids)}"
            )
            self._last_log_time = time.time()

        return result
```

**Expected if this is the issue**:

- Call count increases over epochs (more episodes = more sampling)
- Average time increases over epochs (more tasks = more computation)
- Total time becomes significant fraction of epoch time

### Phase 3: Profile LP Score Caching (30 min)

Test if caching score_tasks() helps:

```python
# In metta/cogworks/curriculum/lp_scorers.py:BidirectionalLPScorer
from functools import lru_cache

class BidirectionalLPScorer(LPScorer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._score_cache_valid = False
        self._score_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def invalidate_cache(self):
        super().invalidate_cache()
        self._score_cache_valid = False
        self._score_cache.clear()

    def score_tasks(self, task_ids: list[int]) -> dict[int, float]:
        # Quick cache check
        cache_key = tuple(sorted(task_ids))
        if self._score_cache_valid and cache_key in self._score_cache:
            self._cache_hits += 1
            return self._score_cache[cache_key]

        self._cache_misses += 1
        result = self._compute_scores(task_ids)  # Existing logic

        # Cache result
        self._score_cache[cache_key] = result
        self._score_cache_valid = True

        # Log cache effectiveness
        if (self._cache_hits + self._cache_misses) % 1000 == 0:
            hit_rate = self._cache_hits / (self._cache_hits + self._cache_misses)
            logger.info(f"LP score cache hit rate: {hit_rate:.1%}")

        return result
```

**Expected if caching helps**:

- High cache hit rate (>90%)
- Significant SPS improvement
- Less degradation over time

### Phase 4: Checkpoint Resume Test (30 min)

Confirm whether resuming from checkpoint resets the performance degradation:

```bash
# Run for 50 epochs, save checkpoint
timeout 300s uv run ./tools/run.py experiment.cogs_v_clips use_lp=True run=perf_checkpoint_1

# Resume from epoch 50, watch if SPS is back to epoch-1 levels
uv run ./tools/run.py experiment.cogs_v_clips use_lp=True run=perf_checkpoint_2 \
  resume_from=<checkpoint_path>
```

**If SPS resets**: Problem is in-memory state accumulation (not in checkpoint) **If SPS stays degraded**: Problem is in
checkpoint data structure

### Phase 5: Isolate Components (60 min)

If performance_mode doesn't fix it, test components individually:

#### 5.1: Disable Gini Only

```python
# In curriculum.py:calculate_gini_coefficients()
def calculate_gini_coefficients(self) -> Dict[str, float]:
    return {}  # Force disable
```

#### 5.2: Disable Per-Label Stats

```python
# In curriculum.py:calculate_per_label_mean_lp_stats()
def calculate_per_label_mean_lp_stats(self) -> Dict[str, float]:
    return {}  # Force disable
```

#### 5.3: Profile Task Scanning

Count how many times `get_all_tracked_tasks()` is called per epoch:

```python
# In task_tracker.py:get_all_tracked_tasks()
def get_all_tracked_tasks(self) -> List[int]:
    if not hasattr(self, '_scan_count'):
        self._scan_count = 0
    self._scan_count += 1
    if self._scan_count % 100 == 0:
        logger.warning(f"get_all_tracked_tasks called {self._scan_count} times")
    # ... existing code
```

### Phase 6: Memory Leak Detection (30 min)

Check for accumulating data structures:

```python
# Add to curriculum.py or learning_progress_algorithm.py
import sys

def _debug_memory_usage(self):
    """Log sizes of major data structures."""
    sizes = {
        'tasks': sys.getsizeof(self._tasks),
        'task_ids': sys.getsizeof(self._task_ids),
        'label_sampling_counts': sys.getsizeof(self._label_sampling_counts),
        'label_eviction_counts': sys.getsizeof(self._label_eviction_counts),
    }
    if hasattr(self._algorithm, 'task_tracker'):
        tracker = self._algorithm.task_tracker
        sizes['task_id_to_index'] = sys.getsizeof(tracker._task_id_to_index)
        sizes['label_hash_to_string'] = sys.getsizeof(tracker._label_hash_to_string)

    logger.info(f"Memory usage (bytes): {sizes}")
```

### Phase 7: Compare with Main (30 min)

Run same tests on main branch to identify regression:

```bash
git stash
git checkout main
# Run Phase 1 tests
git checkout msb_curr_min_v1
git stash pop
```

Compare:

- SPS degradation rate
- Stats collection times
- Memory usage patterns

## Quick Wins to Try (UPDATED - Target the Bottleneck)

### Fix #1: ❌ Performance Mode (DOESN'T HELP - already tested)

### Fix #2: 🎯 Batch Cache Invalidation (LIKELY BEST FIX)

**Problem**: `scorer.invalidate_cache()` called every episode → triggers `get_all_tracked_tasks()` scan **Solution**:
Only invalidate every N episodes

```python
# In metta/cogworks/curriculum/learning_progress_algorithm.py:477-500
def update_task_performance(self, task_id: int, score: float) -> None:
    # Atomic update
    self.task_tracker.update_task_performance_with_bidirectional_emas(
        task_id=task_id,
        score=score,
        scorer=self.scorer if hasattr(self.scorer, "config") else None,
    )

    # MODIFIED: Batch invalidations instead of per-episode
    if not hasattr(self, '_updates_since_invalidation'):
        self._updates_since_invalidation = 0

    self._updates_since_invalidation += 1

    # Invalidate every 100 episodes (tune this number)
    if self._updates_since_invalidation >= 100:
        self.scorer.invalidate_cache()
        self._updates_since_invalidation = 0
    # OLD: self.scorer.invalidate_cache()  # Was called EVERY episode

    self.invalidate_cache()
```

**Expected**:

- Reduces `get_all_tracked_tasks()` calls from ~5000/epoch to ~50/epoch
- Trades slightly stale LP scores for 100x fewer memory scans
- Should see immediate SPS improvement

### Fix #3: Cache Active Task List (COMPLEMENTARY)

**Problem**: `get_all_tracked_tasks()` scans O(max_tasks) slots every time **Solution**: Cache the active task list,
invalidate on task creation/eviction only

```python
# In metta/cogworks/curriculum/task_tracker.py
def __init__(self, ...):
    # ... existing init ...
    self._cached_active_tasks = None
    self._cache_valid = False

def get_all_tracked_tasks(self) -> List[int]:
    # Return cached list if valid
    if self._cache_valid and self._cached_active_tasks is not None:
        return self._cached_active_tasks.copy()

    # Rebuild cache
    if isinstance(self._backend, SharedMemoryBackend):
        task_ids = []
        for i in range(self._backend.max_tasks):
            state = self._backend.get_task_state(i)
            if bool(state.is_active):
                task_ids.append(int(state.task_id))
        self._cached_active_tasks = task_ids
        self._cache_valid = True
        return task_ids.copy()
    else:
        return list(self._task_id_to_index.keys())

def track_task_creation(self, task_id, ...):
    # ... existing logic ...
    self._cache_valid = False  # Invalidate cache

def remove_task(self, task_id):
    # ... existing logic ...
    self._cache_valid = False  # Invalidate cache
```

**Expected**:

- First call per epoch is O(max_tasks)
- Subsequent calls are O(1) cache hits
- Combined with Fix #2, reduces to ~50 scans/epoch instead of 5000

### Fix #4: Reduce Task Pool (TEST ONLY - confirms hypothesis)

```python
# In recipes/experiment/cogs_v_clips.py:282
cur_alg = LearningProgressConfig(
    num_active_tasks=100,  # vs default 256-1000
)
```

**Expected**: If small pool is much faster, confirms O(max_tasks) scaling issue **Tradeoff**: Less curriculum diversity
(only for testing)

## Debugging Tools

### Monitor SPS in Real-Time

```bash
# In separate terminal while training
watch -n 1 'tail -n 50 outputs/*/logs/train.log | grep "SPS:"'
```

### Profile with py-spy

```bash
# Record for 60 seconds during training
py-spy record -o profile.svg --pid $(pgrep -f "tools/run.py")
```

### Memory profiling

```bash
# Use memory_profiler
uv add memory_profiler
# Add @profile decorator to suspect methods
python -m memory_profiler <script>
```

## Expected Outcomes

| Hypothesis       | Test                          | Expected if True       | Next Steps                              |
| ---------------- | ----------------------------- | ---------------------- | --------------------------------------- |
| Stats overhead   | performance_mode=True         | SPS stays constant     | Production fix: enable performance_mode |
| Gini calculation | Disable Gini only             | SPS stays constant     | Optimize or reduce Gini frequency       |
| Memory scanning  | Profile get_all_tracked_tasks | High call count        | Cache results or reduce calls           |
| Lock contention  | Profile with py-spy           | Lock wait time visible | Reduce lock scope or frequency          |
| Memory leak      | Memory profiling              | Unbounded growth       | Fix leak                                |

## Key Files Reference

### Configuration

- `recipes/experiment/cogs_v_clips.py:275-288` - Where LP config is created
- `metta/cogworks/curriculum/learning_progress_algorithm.py:55-252` - LearningProgressConfig

### Stats Collection (Hot Path)

- `metta/rl/training/stats_reporter.py:559-611` - Main stats collection
- `metta/cogworks/curriculum/curriculum.py:529-589` - Curriculum base stats
- `metta/cogworks/curriculum/learning_progress_algorithm.py:575-606` - Algorithm base stats

### Expensive Operations (Performance Impact)

- `metta/cogworks/curriculum/curriculum.py:391-454` - Per-label LP stats
- `metta/cogworks/curriculum/curriculum.py:456-504` - Raw LP debug stats
- `metta/cogworks/curriculum/curriculum.py:506-527` - Gini calculation entry
- `metta/cogworks/curriculum/learning_progress_algorithm.py:652-806` - Gini implementation

### Task Tracking (Potential Bottleneck)

- `metta/cogworks/curriculum/task_tracker.py:490-510` - get_all_tracked_tasks (full scan)
- `metta/cogworks/curriculum/task_tracker.py:238-360` - EMA updates (locking)

### Shared Memory Backend

- `metta/cogworks/curriculum/shared_memory_backend.py:270-395` - SharedMemoryBackend

## Success Criteria

1. **Identify root cause**: Know which component(s) cause degradation
2. **Quantify impact**: Measure SPS improvement from fixes
3. **Production fix**: Enable performance_mode or optimize bottleneck
4. **Verify**: Confirm SPS stays constant over 100+ epochs with LP enabled

## Timeline Estimate

- **Phase 1-2** (Quick check): 45 minutes
- **Phase 3-4** (Deep dive): 60 minutes
- **Phase 5-6** (Isolation): 90 minutes
- **Phase 7** (Comparison): 30 minutes
- **Total**: ~3.5 hours for complete investigation

Start with Phase 1-2 for quick wins, then proceed only if needed.
