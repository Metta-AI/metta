# Curriculum Refactoring Plan

**Branch:** `msb_currcent`
**Date:** 2025-09-30

## Overview
This document tracks the refactoring of the curriculum system to improve performance, clarity, and observability.

## Tasks

### ✅ 1. Remove Seed History Loop
**Status:** ✅ COMPLETED
**Priority:** High (Quick win, no dependencies)

**Goal:** Remove any code that prevents resampling the same seeds

**Current State:**
- No `while seed not in history` pattern found in current codebase
- May have been previously removed or is in a different location

**Changes:**
- [x] Search for seed history prevention code
- [x] Remove if found
- [x] Verify seed generation still works correctly

**Result:**
- Pattern not found in current codebase
- No changes needed

**Files Modified:**
- None (pattern not found in current code)

---

### ✅ 2. Remove Defensive Locking
**Status:** ✅ COMPLETED
**Priority:** High (Performance improvement)

**Goal:** Remove unnecessary locks, keep only critical read-modify-write sections

**Critical Sections (MUST KEEP LOCKS):**
1. `update_task_performance` (lines 477-533) - read-modify-write operation
2. `track_task_creation` (lines 425-457) - writes new task data
3. `load_state` (lines 673-708) - bulk write operation

**Non-Critical Sections (REMOVE LOCKS):**
1. `get_task_stats` (lines 545-593) - read-only, stale data acceptable
2. `get_all_tracked_tasks` (lines 595-598) - read-only, snapshot is fine
3. `get_global_stats` (lines 609-634) - read-only, approximate stats okay
4. `get_state` (lines 636-671) - read-only, checkpoint can be slightly stale

**Maybe Keep:**
1. `update_lp_score` (lines 535-543) - single write, but conflicts with update_task_performance
2. `remove_task` (lines 600-607) - write operation, but rare
3. `_rebuild_task_mapping` (lines 398-416) - internal consistency critical

**Rationale:**
- Stats queries don't need perfect consistency
- Small chance of reading mid-update is acceptable vs lock contention
- Performance over perfect accuracy for metrics

**Result:**
Removed locks from:
- `get_task_stats()` - read-only operation
- `get_all_tracked_tasks()` - snapshot is fine
- `get_global_stats()` - approximate stats acceptable
- `get_state()` - checkpoint can be slightly stale

Kept locks for:
- `update_task_performance()` - critical read-modify-write
- `track_task_creation()` - writes new data
- `load_state()` - bulk write operation
- `remove_task()` - write operation
- `update_lp_score()` - single field update

**Testing:**
- [ ] Run multi-process training to verify no deadlocks
- [ ] Check that stats are reasonable (not corrupted)
- [ ] Monitor for any race condition crashes

**Files Modified:**
- [x] `metta/cogworks/curriculum/task_tracker.py`

---

### ✅ 3. Add Per-Label Learning Progress Logging
**Status:** ✅ COMPLETED
**Priority:** High (Validation & observability)

**Goal:** Log learning progress metrics grouped by task_label (similar to per_label_rewards)

**Pattern to Follow:**
```python
# In mettagrid_env.py:
self.per_label_rewards[self.mg_config.label] = episode_rewards.mean()
infos["per_label_rewards"] = self.per_label_rewards
```

**Implementation Steps:**
1. [x] Add label tracking to CurriculumTask
2. [x] Pass label through curriculum workflow
3. [x] Collect LP scores by label in CurriculumEnv
4. [x] Add to infos dict for logging
5. [ ] Test with existing recipes

**Changes Made:**
- Added `_label` attribute to `CurriculumTask` (extracted from env_cfg.label)
- Added `get_label()` method to `CurriculumTask`
- Added `_per_label_lp_scores` and `_per_label_completion_counts` dicts to `CurriculumEnv`
- Update both dicts on task completion in `step()` method
- Use exponential moving average (alpha=0.1) for LP scores per label
- Added metrics to `_add_curriculum_stats_to_info()` for logging

**Metrics Added:**
- `per_label_lp_scores` - EMA of learning progress scores by task label
- `per_label_completion_counts` - Number of completions per task label

**Files Modified:**
- [x] `metta/cogworks/curriculum/curriculum.py`
- [x] `metta/cogworks/curriculum/curriculum_env.py`

---

### 📋 4. Create Clean Shared Memory API
**Status:** PENDING
**Priority:** Medium (Architecture improvement, but larger refactor)

**Goal:** Abstract shared memory details behind clean interface

**Design:**
```python
class TaskMemoryBackend(ABC):
    """Abstract interface for task memory storage."""

    @abstractmethod
    def allocate(self, max_tasks: int) -> None:
        """Allocate storage for max_tasks."""
        pass

    @abstractmethod
    def read_task(self, task_id: int) -> Optional[TaskData]:
        """Read task data."""
        pass

    @abstractmethod
    def write_task(self, task_id: int, data: TaskData) -> None:
        """Write task data."""
        pass

    @abstractmethod
    def get_all_task_ids(self) -> List[int]:
        """Get all tracked task IDs."""
        pass

class LocalMemoryBackend(TaskMemoryBackend):
    """In-memory implementation."""
    pass

class SharedMemoryBackend(TaskMemoryBackend):
    """Shared memory implementation."""
    pass
```

**Factory Pattern:**
```python
def create_memory_backend(config: LearningProgressConfig) -> TaskMemoryBackend:
    if config.use_shared_memory:
        return SharedMemoryBackend(config.max_memory_tasks, config.session_id)
    else:
        return LocalMemoryBackend(config.max_memory_tasks)
```

**Migration Path:**
1. Create new backend interface and implementations
2. Refactor TaskTracker to use backend
3. Update LearningProgressAlgorithm to use TaskTracker only
4. Remove direct shared memory access from algorithm
5. Test thoroughly with both backends

**Files to Create:**
- `metta/cogworks/curriculum/memory_backend.py` (new)

**Files to Modify:**
- `metta/cogworks/curriculum/task_tracker.py`
- `metta/cogworks/curriculum/learning_progress_algorithm.py`
- `metta/cogworks/curriculum/shared_memory_backend.py` (may deprecate)

---

### 📋 5. Create Impossible Tasks Recipe
**Status:** PENDING
**Priority:** Low (Validation only)

**Goal:** Create assembler chains recipe with impossible tasks to validate LP ignores them

**Recipe Design:**
```python
curriculum_args = {
    "train_with_impossible": {
        "num_agents": [1, 2],
        "chain_lengths": [2, 3, 4],  # Normal tasks
        "num_sinks": [0, 1],
        "room_sizes": ["small", "medium"],
        "positions": [["Any"], ["Any", "Any"]],
        "impossible_ratio": 0.1,  # 10% impossible tasks
    }
}

def make_impossible_task(cfg: MettaGridConfig) -> MettaGridConfig:
    """Create task that always returns 0 reward."""
    # Option 1: Require non-existent resource
    cfg.game_objects.append(create_assembler(
        input_resources={"unobtainium": 1},
        output_resources={"heart": 1}
    ))
    cfg.label += "_IMPOSSIBLE"
    return cfg
```

**Validation Metrics:**
- `per_label_samples[*_IMPOSSIBLE]` > 0 (task was sampled)
- `per_label_rewards[*_IMPOSSIBLE]` ≈ 0 (always fails)
- `per_label_lp_scores[*_IMPOSSIBLE]` < 0.1 (low learning progress)
- `per_label_lp_scores[normal_tasks]` > per_label_lp_scores[impossible_tasks]

**Implementation:**
1. Extend `AssemblyLinesTaskGenerator` with impossible task generation
2. Add `impossible_ratio` config parameter
3. Modify `_generate_task` to occasionally create impossible variant
4. Run short training run and verify metrics

**Files to Create:**
- `experiments/recipes/in_context_learning/assemblers/assembly_lines_with_impossible.py`

**Files to Modify:**
- None (new recipe file only)

---

## Testing Strategy

### Unit Tests
- [x] Verify locking changes don't break single-process mode
- [ ] Test label tracking through curriculum workflow
- [ ] Test memory backend interface implementations

### Integration Tests
- [ ] Multi-process training with shared memory (locking changes)
- [ ] Verify per-label metrics appear in WandB logs
- [ ] Run impossible tasks recipe and verify expected behavior

### Performance Tests
- [ ] Compare SPS before/after locking changes
- [ ] Monitor lock contention in multi-process setup

---

## Rollback Plan

If issues arise:
1. **Locking changes:** Revert specific lock removals if race conditions occur
2. **Label logging:** Feature flag to disable if it causes performance issues
3. **Shared memory API:** Complete refactor can be done on separate branch

---

## Success Criteria

- [ ] No performance regression (SPS maintained or improved)
- [ ] Per-label LP metrics appear in training logs
- [ ] Multi-process training stable with reduced locking
- [ ] Impossible tasks validation shows correct curriculum behavior
- [ ] Code is cleaner and more maintainable

---

## Notes

### Seed History Loop
- Pattern not found in current codebase
- May have been removed in previous refactor
- No action needed unless found elsewhere

### Lock Audit Results
- Identified 10 lock acquisitions in CentralizedTaskTracker
- 3 are critical (must keep)
- 4 are read-only (safe to remove)
- 3 need careful consideration

### Label Implementation Details
- Labels come from `mg_config.label` (set in task generator)
- Need to store in CurriculumTask and pass to env
- Use same aggregation pattern as rewards (moving average)
