# Learning Progress Eviction Performance Fix

## The Problem

Performance testing showed that **none of the proposed solutions** (batch invalidation, task list caching, different batch sizes) prevented the performance decay from ~45k SPS to ~20k SPS over time.

## Root Cause Analysis

The real bottleneck was **O(n²) complexity in the eviction logic**, not cache invalidation frequency.

### The Catastrophic Code Path

Every time a task needed to be evicted (which happens frequently when the pool is full at 1000 tasks):

**`recommend_eviction()` (line 342):**
```python
evictable_tasks = [tid for tid in all_task_ids if self.should_evict_task(tid, min_presentations)]
```

This called `should_evict_task()` for **EACH of the 1000 tasks** in the pool.

**`should_evict_task()` (lines 364-368):**
```python
all_task_ids = self.task_tracker.get_all_tracked_tasks()  # Line 364
scores = self.score_tasks(all_task_ids)  # Line 368 - scores ALL 1000 tasks!
```

Each call did:
1. `get_all_tracked_tasks()` - scans shared memory for all 1000 tasks
2. `score_tasks(all_task_ids)` - calculates LP scores for all 1000 tasks

### The Math

With a full pool of 1000 tasks:
- **Per eviction**: 1000 calls to `should_evict_task()`
- **Per call**: Calculate scores for 1000 tasks
- **Total**: ~1,000,000 score calculations per eviction
- **Complexity**: O(n²) where n = pool size

As the pool grew from 0 to 1000 tasks over time, eviction cost grew quadratically, causing the performance decay.

### Additional Hot Path Issue

**`on_task_evicted()` (lines 420-422):**
```python
for tid in self.task_tracker.get_all_tracked_tasks():
    label = self.task_tracker.get_task_label(tid)
    all_active_labels.add(label)
```

This scanned all 1000 tasks on **every single eviction** just to check if a label was still active.

## The Fix

### 1. Refactored `recommend_eviction()` to O(n)

**Before:**
- Called `should_evict_task()` 1000 times
- Each call calculated scores for all 1000 tasks
- Result: O(n²) complexity

**After:**
- Calculate scores **once** for all tasks
- Calculate threshold **once**
- Loop through tasks with pre-computed scores
- Result: O(n) complexity

```python
def recommend_eviction(self, all_task_ids: List[int], min_presentations: int) -> Optional[int]:
    if not all_task_ids:
        return None

    # PERFORMANCE FIX: Calculate scores ONCE for all tasks
    all_scores = self.score_tasks(all_task_ids)  # O(n)

    # Calculate threshold once
    sorted_scores = sorted(all_scores.values())  # O(n log n)
    threshold_index = max(0, int(len(sorted_scores) * self.hypers.eviction_threshold_percentile))
    threshold_score = sorted_scores[threshold_index] if sorted_scores else 0.0

    # Filter to evictable tasks using pre-computed scores
    evictable_tasks = []
    for tid in all_task_ids:  # O(n)
        task_stats = self.task_tracker.get_task_stats(tid)
        if task_stats is None or task_stats["completion_count"] < min_presentations:
            continue

        task_score = all_scores.get(tid, 0.0)
        if task_score <= threshold_score:
            evictable_tasks.append(tid)

    if not evictable_tasks:
        return None

    # Find task with minimum learning progress
    min_task_id = min(evictable_tasks, key=lambda tid: all_scores.get(tid, 0.0))
    return min_task_id
```

### 2. Removed Expensive Label Tracking from Hot Path

**Before:**
```python
def on_task_evicted(self, task_id: int):
    # ... eviction logic ...

    # Scan ALL tasks on every eviction
    all_active_labels = set()
    for tid in self.task_tracker.get_all_tracked_tasks():  # O(n)
        label = self.task_tracker.get_task_label(tid)
        all_active_labels.add(label)
```

**After:**
```python
def on_task_evicted(self, task_id: int):
    # ... eviction logic ...

    # PERFORMANCE FIX: Don't scan on every eviction
    # Label tracking done lazily during stats collection
    self._active_labels.discard(evicted_label)
    if evicted_label not in self._inactive_labels_fifo:
        self._inactive_labels_fifo.append(evicted_label)
```

The active labels set is now maintained in `get_pool_composition_stats()` which already scans all tasks for other purposes (called once per epoch, not per eviction).

### 3. Removed Deprecated Method

Removed the old `should_evict_task()` method entirely since its logic is now inlined in `recommend_eviction()` with proper O(n) complexity.

## Expected Impact

### Performance Improvement

- **Eviction cost**: O(n²) → O(n)
- **With 1000 tasks**: ~1,000,000 operations → ~1,000 operations
- **Speedup**: ~1000x faster eviction

### When It Matters

- **Early training**: Small pool, minimal impact
- **Mid-late training**: Large pool (500-1000 tasks), **massive** impact
- **Steady state**: Full pool with frequent evictions, **critical** impact

This explains why performance was good initially (~45k SPS) but degraded over time as the pool filled up.

## Testing

Re-run the performance tests with this fix:

```bash
PREFIX=msb_perfdiagnosis_v2_ ./tools/launch_lp_perf_matrix.sh quick
```

Expected result:
- Performance should remain stable at ~45k SPS throughout training
- No degradation as pool size increases
- All configurations should perform similarly (since the bottleneck is fixed)

## Files Changed

- `metta/cogworks/curriculum/learning_progress_algorithm.py`:
  - Refactored `recommend_eviction()` from O(n²) to O(n)
  - Simplified `on_task_evicted()` to avoid hot-path scanning
  - Removed deprecated `should_evict_task()` method
  - Updated `get_pool_composition_stats()` to maintain active labels

## Commits

- `fix: eliminate O(n²) complexity in LP eviction logic`

