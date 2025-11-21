# Curriculum System Issues - Planning Document

## Overview

This document explores identified issues in the curriculum system, particularly around state persistence, shared memory, and cross-process coordination. Issues are organized by file and include investigation steps and potential solutions.

---

## 1. `learning_progress_algorithm.py`

### Issue 1.1: Checkpoint Load Ignores Persisted Hyperparameters
**Confidence: 4/5**

#### Problem
The `load_state` method ignores persisted `hypers`, allowing resumed runs to mix checkpointed shared-memory layout with new configs. This can cause mismatches in:
- `task_struct_size`
- `num_active_tasks`
- `session_id`
- Eviction thresholds

#### Investigation Steps
1. Review `load_state` implementation in `learning_progress_algorithm.py`
2. Check what hyperparameters affect shared memory layout
3. Identify which configs can safely change vs. which must match checkpoint
4. Review how `save_state` serializes hypers

#### Potential Solutions
- **Option A**: Validate loaded hypers match current config, error if incompatible
- **Option B**: Rehydrate from saved hypers, overriding config file
- **Option C**: Version checkpoint format, allow migration logic
- **Option D**: Document which params must not change between checkpoint/resume

#### Questions
- Are there legitimate use cases for changing hyperparameters on resume?
- Should we prioritize backward compatibility or correctness?
- What's the failure mode - silent corruption or crashes?

---

### Issue 1.2: Label Strings Lost on Checkpoint Resume
**Confidence: 4/5**

#### Problem
Checkpoint load restores tasks but drops label strings (only label hashes exist in shared memory). After resume:
- Per-label stats become empty
- Gini coefficient calculations fail
- Label-based sampling breaks

#### Investigation Steps
1. Examine shared memory struct - confirm it only stores `label_hash` not strings
2. Review `save_state` - check if labels are serialized
3. Check `load_state` - verify label→hash mapping restoration
4. Test checkpoint/resume cycle to confirm label loss

#### Potential Solutions
- **Option A**: Persist hash→string mapping in checkpoint, reload in `load_state`
- **Option B**: Store label strings directly in shared memory (increases size)
- **Option C**: Rebuild labels from task configs after load (if deterministic)
- **Option D**: Use external label registry that persists independently

#### Questions
- How critical is per-label tracking vs. per-task tracking?
- Can we reconstruct labels from task configs reliably?
- What's the memory overhead of storing strings in shared memory?

---

## 2. `task_tracker.py`

### Issue 2.1: Stale Label Hashes in Freed Slots
**Confidence: 4/5**

#### Problem
Freed task slots retain stale `label_hash` values. New tasks reusing the slot inherit incorrect labels, corrupting per-label statistics.

#### Investigation Steps
1. Review task creation/removal logic in `task_tracker.py`
2. Check which fields are zeroed on slot release (note: should zero index 17)
3. Verify label_hash field location in struct
4. Write test that creates, removes, and re-creates tasks in same slot

#### Potential Solutions
- **Option A**: Zero `label_hash` field (index 17) on task removal
- **Option B**: Zero entire slot on removal (safer but may be overkill)
- **Option C**: Add validation to reject stale label_hash values
- **Option D**: Use task generation counter to detect stale slots

#### Questions
- Which fields MUST be zeroed vs. which are safely overwritten?
- Is there a "slot version" field we can use?
- Should we have a comprehensive "reset slot" function?

---

### Issue 2.2: Non-Deterministic Python Hash Function
**Confidence: 5/5** ✅ **FIXED**

#### Problem
Uses Python's `hash(label)` which is randomized per process. This causes:
- Label hashes differ across worker processes
- Workers can't resolve labels from other processes
- Hash collisions become more likely
- Cross-process label tracking is broken

**SOLUTION IMPLEMENTED**: Replaced Python's `hash()` with SHA256-based deterministic hash in `task_tracker.py:set_task_label()`. All processes now compute identical hashes for the same label string. See `outputs/deterministic_hash_fix_summary.md` for details.

#### Investigation Steps
1. Confirm `hash()` is used for label hashing
2. Review where label_hash is read/written across processes
3. Check if workers need to resolve labels from other workers
4. Test hash consistency across multiple Python processes

#### Potential Solutions
- **Option A**: Use deterministic hash (SHA1, xxhash, CRC32)
- **Option B**: Truncate to 53 bits (safe integer range) if using numeric hash
- **Option C**: Store labels directly in shared memory instead of hashes
- **Option D**: Maintain centralized label→hash registry

#### Questions
- What's the acceptable collision rate for label hashes?
- Is 53-bit hash space sufficient for expected label count?
- Should we support label enumeration/listing?

---

### Issue 2.3: Label Readers Can't See Cross-Process Tasks
**Confidence: 4/5**

#### Problem
Label readers rely on local `_task_id_to_index` mapping. Tasks created by other workers aren't visible locally, so per-label stats miss cross-process tasks.

#### Investigation Steps
1. Review how `_task_id_to_index` is populated
2. Check if shared memory backend syncs this mapping
3. Identify all label-based queries (stats, sampling, etc.)
4. Test multi-process scenario: process A creates task, process B queries by label

#### Potential Solutions
- **Option A**: Rebuild mapping by scanning shared memory on each query
- **Option B**: Maintain mapping in shared memory alongside tasks
- **Option C**: Use label_hash directly to scan all task slots
- **Option D**: Accept eventually-consistent label view, rebuild periodically

#### Questions
- How often are label-based queries performed?
- Is scan-on-query performance acceptable?
- Can we cache mapping with invalidation strategy?

---

### Issue 2.4: Label Mapping Not Restored on Checkpoint Load
**Confidence: 4/5**

#### Problem
`load_state` writes `label_hash` back to shared memory but never restores `_label_hash_to_string` mapping. All labels appear as `None` after checkpoint resume.

#### Investigation Steps
1. Review `save_state` - confirm it saves label mapping
2. Review `load_state` - confirm it doesn't restore mapping
3. Check what breaks when labels are None (stats? sampling?)
4. Test checkpoint/resume to observe label behavior

#### Potential Solutions
- **Option A**: Restore `_label_hash_to_string` dict in `load_state`
- **Option B**: Include label mapping in checkpoint format
- **Option C**: Rebuild mapping from loaded task configs
- **Option D**: Store labels in shared memory (eliminates need for mapping)

#### Questions
- Is this the same issue as 1.2, just from different angle?
- Should task_tracker own label persistence or learning_progress_algorithm?
- Can we unify label storage strategy across components?

---

## 3. `shared_memory_backend.py`

### Issue 3.1: Lock Not Actually Shared Across Processes
**Confidence: 5/5** ✅ **CONFIRMED**

#### Problem
Each process recreates its own `Lock` after pickling. The lock isn't shared across processes, so concurrent writers race on shared memory without synchronization.

**Root Cause Analysis (COMPLETED):**
1. `SharedMemoryBackend.__init__` creates `self._lock = Lock()` at line 189
2. When curriculum is pickled to send to worker processes (via PufferLib Multiprocessing):
   - `__getstate__` (line 280) doesn't include the lock in state dict
   - `__setstate__` (line 289) recreates a NEW lock with `self._lock = Lock()` at line 309
3. Result: Each process has its own independent `multiprocessing.Lock()` object
4. These locks protect nothing - each process thinks it has exclusive access

**Critical Operations Racing Without Synchronization:**
- Task creation (`track_task_creation`)
- Task removal (`remove_task`)
- Performance updates (reward/LP score writes)
- Label hash updates
- State loading (clearing/repopulating shared memory)
- Task mapping rebuilds

**Observed in Code:**
- Line 189: Initial lock creation `self._lock = Lock()`
- Line 309: Lock recreated after pickle `self._lock = Lock()`
- Line 232-239: `acquire_lock()` returns the local lock (useless across processes)
- Comment at line 308 is misleading: "Recreate lock (each process needs its own lock object pointing to the shared lock)" - it actually creates an independent lock

#### Potential Solutions
- **Option A**: Use `multiprocessing.Manager().Lock()` for shared lock ⭐ **RECOMMENDED**
  - Manager creates server process that coordinates lock across workers
  - Lock object can be pickled and survives process boundaries
  - Standard pattern for shared locks in multiprocessing
  - Performance: Small overhead but necessary for correctness

- **Option B**: Create Manager in parent, pass lock to backend constructor
  - Requires changing TaskTracker/LearningProgressAlgorithm initialization
  - More invasive but potentially cleaner architecture

- **Option C**: Use `multiprocessing.synchronize.Lock` with explicit passing
  - Similar to Option B, requires parent to create and pass lock

- **Option D**: Store lock handle in shared memory segment
  - Complex, error-prone, not recommended

- **Option E**: Use file-based locking (fcntl/flock)
  - Works but slower than Manager lock
  - Good fallback for certain platforms

#### Recommended Solution Details

**Option A Implementation:**
```python
# In SharedMemoryBackend.__init__:
from multiprocessing import Manager
self._manager = Manager()
self._lock = self._manager.Lock()

# In __getstate__:
return {
    ...,
    "lock": self._lock,  # Manager locks can be pickled
}

# In __setstate__:
self._lock = state["lock"]  # Use the shared lock
```

This works because Manager.Lock() returns a proxy object that can be pickled and references the same server-backed lock across all processes.

#### Questions
- ✅ What operations need locking? **Answer: All shared memory writes (task creation, updates, removal)**
- ✅ Is there a performance cost to using Manager lock? **Answer: Yes, but small and necessary for correctness**
- ❌ Can we use lock-free algorithms? **Answer: Not easily - multi-field updates need atomicity**

---

### Issue 3.2: Shared Memory Leaks on Abnormal Exit
**Confidence: 3/5**

#### Problem
`__del__` only closes shared memory, never unlinks it. If process crashes or `cleanup()` isn't called, shared memory segments leak.

#### Investigation Steps
1. Review `__del__` and `cleanup` implementations
2. Check who should be responsible for unlinking (creator only?)
3. Test abnormal exit scenarios (Ctrl+C, exception, segfault)
4. List shared memory segments before/after runs (`/dev/shm` on Linux)

#### Potential Solutions
- **Option A**: Track creator and unlink in `__del__` if creator
- **Option B**: Register `atexit` handler to call cleanup
- **Option C**: Use reference counting across processes
- **Option D**: Provide cleanup utility script
- **Option E**: Accept leak, document manual cleanup steps

#### Questions
- Should we prioritize robustness over performance?
- Is there a pattern from multiprocessing best practices?
- What happens if non-creator unlinks while others are using it?

---

## 4. `lp_scorers.py`

### Issue 4.1: Zero-Count LP Distribution Metric Broken
**Confidence: 5/5**

#### Problem
`num_zeros_lp_dist` uses `lp_scores == 0` on a Python list, which always evaluates to `False`. The zero-count metric never works correctly.

#### Investigation Steps
1. Review `num_zeros_lp_dist` implementation
2. Confirm `lp_scores` is a list not numpy array
3. Check where this metric is used/logged
4. Verify if anyone relies on this metric

#### Potential Solutions
- **Option A**: Convert to numpy: `np.sum(np.array(lp_scores) == 0)`
- **Option B**: List comprehension: `sum(1 for x in lp_scores if x == 0)`
- **Option C**: Use `lp_scores.count(0)` if it's a list
- **Option D**: Track zero-count separately during score calculation

#### Questions
- Is this metric actually used/monitored?
- Should we remove it if unused?
- Are there other similar bugs from list/array confusion?

---

### Issue 4.2: Distribution Normalization Includes Ineligible Tasks
**Confidence: 3/5**

#### Problem
`_calculate_task_distribution` normalizes over all tasks, including those under `min_samples_for_lp` threshold. These tasks later bypass LP sampling (get exploration bonus instead), diluting probabilities for eligible tasks.

#### Investigation Steps
1. Review `_calculate_task_distribution` logic
2. Check how `min_samples_for_lp` threshold is applied
3. Trace sampling flow: distribution calculation → task selection
4. Measure impact: what % of tasks are typically ineligible?

#### Potential Solutions
- **Option A**: Exclude tasks under threshold from normalization
- **Option B**: Normalize then re-normalize after filtering
- **Option C**: Use separate exploration vs. exploitation distributions
- **Option D**: Accept dilution as intended behavior (exploration boost)

#### Questions
- Is this a bug or feature (implicit exploration boost)?
- Does this significantly affect curriculum dynamics?
- Should ineligible tasks get explicit exploration weight?

---

## 5. `curriculum_env.py`

### Issue 5.1: Eviction Counters Zeroed Mid-Epoch
**Confidence: 5/5**

#### Problem
`get_and_reset_evictions_this_epoch` is called on every episode completion, zeroing epoch counters mid-epoch. Consequences:
- StatsReporter sees 0 evictions at actual epoch end
- Eviction info in `infos` only reflects last reset
- Eviction tracking is per-episode not per-epoch

#### Investigation Steps
1. Find all calls to `get_and_reset_evictions_this_epoch`
2. Trace when episodes complete vs. when epochs end
3. Review StatsReporter to see what it expects
4. Check WandB logs to see if eviction counts look wrong

#### Potential Solutions
- **Option A**: Move reset to epoch boundary only
- **Option B**: Accumulate evictions in env, let reporter read & reset
- **Option C**: Rename to `get_and_reset_evictions_this_episode` (match behavior)
- **Option D**: Track both per-episode and per-epoch evictions separately

#### Questions
- What's the intended granularity for eviction reporting?
- Who decides when epoch ends - env or trainer?
- Are there other "epoch" metrics with this issue?

---

### Issue 5.2: Failed Task Configs Don't Update Curriculum
**Confidence: 4/5**

#### Problem
When task config fails in `_get_task_with_retries`, we mark task complete with -1.0 but never call `curriculum.update_task_performance`. The curriculum algorithm never sees the failure, so:
- Invalid tasks persist in task pool
- They can be resampled repeatedly
- Their scores/eviction signals never update
- Curriculum learning is disrupted

#### Investigation Steps
1. Review `_get_task_with_retries` exception handling
2. Check if `-1.0` score has special meaning elsewhere
3. Trace what happens after marking complete with -1.0
4. Test with intentionally broken task config

#### Potential Solutions
- **Option A**: Call `update_task_performance` with failure signal before marking complete
- **Option B**: Remove failed tasks from curriculum immediately
- **Option C**: Track failure count, evict after N consecutive failures
- **Option D**: Log failure but let natural eviction handle it

#### Questions
- Should failed configs be retried or permanently removed?
- Is there a difference between transient vs. permanent failures?
- Should failures affect label-level stats?

---

## Investigation Priority

### High Priority (Fix First)
1. ✅ **Lock not shared** (3.1) - FIXED: Now uses Manager().Lock() for proper cross-process synchronization
2. ✅ **Non-deterministic hash** (2.2) - FIXED: Now uses SHA256-based deterministic hash
3. **Eviction counters mid-epoch** (5.1) - High confidence, clear fix
4. **Zero-count metric broken** (4.1) - High confidence, easy fix

### Medium Priority
5. **Failed task configs** (5.2) - Curriculum quality issue
6. **Checkpoint ignores hypers** (1.1) - Silent corruption risk
7. **Label strings lost** (1.2) - Breaks post-checkpoint functionality
8. **Stale label hashes** (2.1) - Data corruption

### Lower Priority (Investigate Further)
9. **Cross-process label visibility** (2.3) - May be acceptable limitation
10. **Label mapping not restored** (2.4) - Overlaps with 1.2
11. **Distribution normalization** (4.2) - Low confidence, may be intended
12. **Shared memory leaks** (3.2) - Lower confidence, operational issue

---

## Testing Strategy

### Unit Tests Needed
- [ ] Checkpoint save/load preserves labels and hypers
- [ ] Task slot properly zeroed on removal
- [ ] Label hash is deterministic across processes
- [ ] Lock actually synchronizes across processes
- [ ] Eviction counters reset at right time
- [ ] Failed task configs trigger curriculum update

### Integration Tests Needed
- [ ] Multi-process curriculum with concurrent task creation
- [ ] Checkpoint/resume preserves full curriculum state
- [ ] Cross-process label-based queries work correctly
- [ ] Abnormal exit cleanup (manual test)

### Manual Tests
- [ ] Run multi-process training, checkpoint, resume
- [ ] Inspect shared memory segments for leaks
- [ ] Verify WandB metrics look correct
- [ ] Intentionally break task configs, observe behavior

---

## Notes

- Several issues are interrelated (label persistence, cross-process coordination)
- May want to refactor shared memory strategy holistically
- Consider whether to prioritize correctness vs. backward compatibility
- Document assumptions about single vs. multi-process usage


