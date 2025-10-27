# Curriculum Logging Migration Plan

**Goal**: Move curriculum stats from per-environment batched logging (choppy) to centralized epoch-level logging (smooth)

**Status**: Ready for implementation

**Key Design Decisions**:
- `per_label_samples_this_epoch` always emitted (not gated by troubleshooting flag) - needed for basic curriculum monitoring
- Tracked task dynamics (Group C) put behind troubleshooting flag to reduce overhead
- StatsReporter has access to curriculum via `self.context.curriculum` (set in trainer from `env._curriculum`)
- `show_curriculum_troubleshooting_logging` flag already exists in `LearningProgressAlgorithmHypers`
- Step 5 is required (not optional) since Step 1 depends on `get_per_label_lp_scores()`

---

## Overview

### Current Architecture (Problematic)
```
Per-Env (50-step batch) → Info Dicts → Accumulation → Averaging → WandB
                                                         ↓
                                                    Choppy graphs
```

### Target Architecture (Clean)
```
Curriculum (shared) → Epoch boundary → StatsReporter → WandB
                                             ↓
                                        Smooth graphs

Per-Episode Counts → Info Dicts → Summed → WandB (only per_label_samples)
```

---

## Logged Quantities Classification

### Group A: Global Curriculum Stats
**Source**: `curriculum.stats()`
**Action**: Move to centralized epoch-level collection

- `total_completions` - Total tasks completed
- `num_evicted` - Total tasks evicted
- `mean_pool_lp_score` - Mean LP score in task pool
- `pool_composition_fraction` - Dict {label: fraction} for pool composition
- `per_label_aggregate_evictions` - Dict {label: cumulative eviction count}

### Group B: Derived Gini Coefficients
**Source**: Computed from `curriculum.stats()`
**Action**: Move to centralized epoch-level collection

- `sampling_gini` - Inequality in sampling distribution
- `pool_occupancy_gini` - Inequality in completion counts
- `pool_lp_gini` - Inequality in LP scores

### Group C: Tracked Task Dynamics (first 3 tasks)
**Source**: Currently tracked in CurriculumEnv
**Action**: Move to centralized AND put behind troubleshooting flag (NEW: gated for performance)

- `tracked_task_lp_scores` - Dict {task_0, task_1, task_2: LP score}
- `tracked_task_completions_this_epoch` - Dict {task_0, task_1, task_2: count}

**Note**: This tracking is now conditional to reduce overhead when not debugging specific tasks.

### Group D: Per-Label Troubleshooting (behind flag)
**Source**: Currently tracked in CurriculumEnv with EMA
**Action**: Move to centralized troubleshooting collection

- `per_label_lp_scores` - Dict {label: raw LP score} (EMA)
- `per_label_postzscored_lp_scores` - Dict {label: post-zscore LP} (EMA)
- `per_label_lp_probs` - Dict {label: sampling probability} (EMA)
- `curriculum_input_rewards/task_slot_{0,1,2}` - List of rewards for tracked tasks

### Group E: Per-Episode Counts (KEEP in info dicts)
**Source**: Emitted on episode completion
**Action**: Keep in info dicts, needs episode-level granularity for proper summing
**Gating**: ALWAYS ENABLED (not behind troubleshooting flag - needed for basic curriculum monitoring)

- `per_label_samples_this_epoch` - Dict {label: 1} (summed across episodes)

**Note**: This is the ONLY curriculum stat that remains in info dicts. All others move to centralized collection.

---

## Implementation Steps

### Step 1: Add Centralized Collection in StatsReporter

**File**: `metta/rl/training/stats_reporter.py`

**Add new method**:
```python
def _collect_curriculum_stats(self) -> dict[str, float]:
    """Collect curriculum statistics directly at epoch boundary.

    This replaces the batched per-environment logging approach with
    centralized collection, providing smooth and consistent logging.
    """
    if not self.context.curriculum:
        return {}

    curriculum = self.context.curriculum
    stats = {}

    # Get base curriculum stats
    curriculum_stats = curriculum.stats()

    # ===== GROUP A: Global Curriculum Stats =====
    if "num_completed" in curriculum_stats:
        stats["curriculum_stats/total_completions"] = float(curriculum_stats["num_completed"])

    if "num_evicted" in curriculum_stats:
        stats["curriculum_stats/num_evicted"] = float(curriculum_stats["num_evicted"])

    if "algorithm/mean_lp_score" in curriculum_stats:
        stats["curriculum_stats/mean_pool_lp_score"] = float(curriculum_stats["algorithm/mean_lp_score"])

    # Pool composition fractions
    total_pool_size = curriculum_stats.get("num_active_tasks", 0)
    if total_pool_size > 0:
        for key, value in curriculum_stats.items():
            if key.startswith("algorithm/pool_composition/"):
                label = key.replace("algorithm/pool_composition/", "")
                stats[f"curriculum_stats/pool_composition_fraction/{label}"] = float(value / total_pool_size)

    # Per-label aggregate evictions
    for key, value in curriculum_stats.items():
        if key.startswith("algorithm/eviction_counts/"):
            label = key.replace("algorithm/eviction_counts/", "")
            stats[f"curriculum_stats/per_label_aggregate_evictions/{label}"] = float(value)

    # ===== GROUP B: Derived Gini Coefficients =====
    # Sampling gini
    sampling_counts = [v for k, v in curriculum_stats.items()
                      if k.startswith("algorithm/sampling_counts/")]
    if sampling_counts:
        stats["curriculum_stats/sampling_gini"] = self._calculate_gini_coefficient(sampling_counts)

    # Pool occupancy gini
    completion_counts = [v for k, v in curriculum_stats.items()
                        if k.startswith("algorithm/completion_counts/")]
    if completion_counts:
        stats["curriculum_stats/pool_occupancy_gini"] = self._calculate_gini_coefficient(completion_counts)

    # Pool LP gini
    lp_scores = [v for k, v in curriculum_stats.items()
                if k.startswith("algorithm/lp_scores/")]
    if lp_scores:
        stats["curriculum_stats/pool_lp_gini"] = self._calculate_gini_coefficient(lp_scores)

    # ===== GROUP C & D: Troubleshooting Stats (if enabled) =====
    if self._should_enable_curriculum_troubleshooting():
        stats.update(self._collect_curriculum_troubleshooting_stats(curriculum, curriculum_stats))

    return stats

def _should_enable_curriculum_troubleshooting(self) -> bool:
    """Check if curriculum troubleshooting logging is enabled.

    Checks the curriculum algorithm's hyperparameters for the
    show_curriculum_troubleshooting_logging flag.

    Context Access: self.context.curriculum is set in trainer from env._curriculum
    Flag Location: LearningProgressAlgorithmHypers.show_curriculum_troubleshooting_logging
    """
    curriculum = self.context.curriculum
    if not curriculum or not hasattr(curriculum, "_algorithm"):
        return False

    algorithm = curriculum._algorithm
    if not algorithm or not hasattr(algorithm, "hypers"):
        return False

    return getattr(algorithm.hypers, "show_curriculum_troubleshooting_logging", False)

def _collect_curriculum_troubleshooting_stats(
    self,
    curriculum: Any,
    curriculum_stats: dict
) -> dict[str, float]:
    """Collect detailed troubleshooting stats for curriculum debugging.

    Includes:
    - Tracked task dynamics (first 3 tasks)
    - Per-label LP scores at different stages
    - Input reward distributions for tracked tasks
    """
    stats = {}

    # GROUP C: Tracked task dynamics
    # Get first 3 task IDs from the curriculum's task pool
    tracked_task_ids = self._get_tracked_task_ids(curriculum)

    if tracked_task_ids:
        for i, task_id in enumerate(tracked_task_ids):
            lp_score = curriculum.get_task_lp_score(task_id)
            stats[f"curriculum_stats/tracked_task_lp_scores/task_{i}"] = float(lp_score)

            # Completion counts this epoch from accumulated info dicts
            completion_key = f"curriculum_stats/tracked_task_completions_this_epoch/task_{i}"
            if completion_key in self._state.rollout_stats:
                stats[completion_key] = float(np.sum(self._state.rollout_stats[completion_key]))

    # GROUP D: Per-label LP scores
    # Collect from curriculum algorithm's per-label tracking
    per_label_stats = self._get_per_label_lp_stats(curriculum)
    stats.update(per_label_stats)

    return stats

def _get_tracked_task_ids(self, curriculum: Any) -> list[int]:
    """Get first 3 task IDs for detailed tracking."""
    # Get active tasks from curriculum
    if not hasattr(curriculum, "_task_pool"):
        return []

    task_pool = curriculum._task_pool
    if not task_pool:
        return []

    # Return first 3 task IDs (or fewer if pool is smaller)
    return list(task_pool.keys())[:3]

def _get_per_label_lp_stats(self, curriculum: Any) -> dict[str, float]:
    """Get per-label LP scores from curriculum algorithm.

    Requires: Step 5 implementation (get_per_label_lp_scores method on algorithm)
    """
    stats = {}

    algorithm = getattr(curriculum, "_algorithm", None)
    if not algorithm:
        return stats

    # Get per-label aggregated scores from algorithm (Step 5 provides this)
    if hasattr(algorithm, "get_per_label_lp_scores"):
        per_label_scores = algorithm.get_per_label_lp_scores()
        for label, score_dict in per_label_scores.items():
            stats[f"curriculum_stats/per_label_lp_scores/{label}"] = float(score_dict.get("raw", 0.0))
            stats[f"curriculum_stats/per_label_postzscored_lp_scores/{label}"] = float(score_dict.get("postzscored", 0.0))
            stats[f"curriculum_stats/per_label_lp_probs/{label}"] = float(score_dict.get("prob", 0.0))

    return stats

@staticmethod
def _calculate_gini_coefficient(values: list[float]) -> float:
    """Calculate Gini coefficient for a distribution.

    Measures inequality in sampling across labels:
    - 0 = perfect equality (all labels sampled equally)
    - 1 = perfect inequality (all samples from one label)
    """
    if not values or len(values) == 0:
        return 0.0

    if sum(values) == 0:
        return 0.0

    sorted_values = sorted(values)
    n = len(sorted_values)

    cumsum = sum((i + 1) * v for i, v in enumerate(sorted_values))
    total = sum(sorted_values)
    gini = (2.0 * cumsum) / (n * total) - (n + 1.0) / n

    return float(gini)
```

**Integrate into `_build_wandb_payload()`**:
```python
def _build_wandb_payload(...) -> dict[str, float]:
    """Convert collected stats into a flat wandb payload."""

    if experience is None:
        return {}

    # Existing processing
    processed = process_training_stats(
        raw_stats=self._state.rollout_stats,
        losses_stats=losses_stats,
        experience=experience,
        trainer_config=trainer_cfg,
    )

    # NEW: Collect curriculum stats at epoch level
    curriculum_stats = self._collect_curriculum_stats()

    timing_info = compute_timing_stats(timer=timer, agent_step=agent_step)
    self._normalize_steps_per_second(timing_info, agent_step)

    weight_stats = self._collect_weight_stats(policy=policy, epoch=epoch)
    system_stats = self._collect_system_stats()
    memory_stats = self._collect_memory_stats()
    parameters = self._collect_parameters(...)
    hyperparameters = self._collect_hyperparameters(...)

    return build_wandb_payload(
        processed_stats=processed,
        curriculum_stats=curriculum_stats,  # NEW
        timing_info=timing_info,
        weight_stats=weight_stats,
        grad_stats=self._state.grad_stats,
        system_stats=system_stats,
        memory_stats=memory_stats,
        parameters=parameters,
        hyperparameters=hyperparameters,
        evals=self._state.eval_scores,
        agent_step=agent_step,
        epoch=epoch,
    )
```

**Update `build_wandb_payload()` signature**:
```python
def build_wandb_payload(
    processed_stats: dict[str, Any],
    curriculum_stats: dict[str, Any],  # NEW
    timing_info: dict[str, Any],
    weight_stats: dict[str, Any],
    grad_stats: dict[str, float],
    system_stats: dict[str, Any],
    memory_stats: dict[str, Any],
    parameters: dict[str, Any],
    hyperparameters: dict[str, Any],
    evals: EvalRewardSummary,
    *,
    agent_step: int,
    epoch: int,
) -> dict[str, float]:
    """Create a flattened stats dictionary ready for wandb logging."""

    # ... existing code ...

    # Add curriculum stats (already has proper prefixes)
    _update(curriculum_stats)

    return payload
```

---

### Step 2: Simplify CurriculumEnv

**File**: `metta/cogworks/curriculum/curriculum_env.py`

**Remove these attributes from `__init__`**:
```python
# DELETE:
self._stats_update_counter = 0
self._stats_update_frequency = 50
self._cached_curriculum_stats = {}
self._curriculum_stats_cache_valid = False
self._per_label_lp_scores = {}
self._per_label_postzscored_lp_scores = {}
self._per_label_lp_probs = {}
self._task_slot_input_rewards = {0: [], 1: [], 2: []}
self._tracked_task_ids = []
self._tracked_task_completions_this_epoch = {}
self._tracked_task_completions_baseline = {}
```

**Keep only**:
```python
# Minimal tracking for troubleshooting (tracked tasks)
self._enable_per_label_tracking = False
if hasattr(curriculum, "_algorithm") and curriculum._algorithm is not None:
    if hasattr(curriculum._algorithm, "hypers"):
        self._enable_per_label_tracking = curriculum._algorithm.hypers.show_curriculum_troubleshooting_logging

# Tracked task attributes (only if troubleshooting enabled)
if self._enable_per_label_tracking:
    self._tracked_task_ids = []
    self._tracked_task_completions_this_epoch = {}
    self._tracked_task_completions_baseline = {}
else:
    self._tracked_task_ids = None
    self._tracked_task_completions_this_epoch = None
    self._tracked_task_completions_baseline = None
```

**Delete these methods entirely**:
```python
# DELETE:
def reset_epoch_counters(self) -> None:
    """No longer needed - rollout stats are cleared by StatsReporter"""
    pass

def _add_curriculum_stats_to_info(self, info_dict: dict) -> None:
    """No longer needed - stats collected centrally"""
    pass

def _calculate_gini_coefficient(self, values: list[float]) -> float:
    """Moved to StatsReporter"""
    pass

def set_stats_update_frequency(self, frequency: int) -> None:
    """No longer needed - no batching"""
    pass

def force_stats_update(self) -> None:
    """No longer needed - no batching"""
    pass
```

**Simplify `reset()` method**:
```python
def reset(self, *args, **kwargs):
    """Reset the environment and get a new task from curriculum."""
    obs, info = self._env.reset(*args, **kwargs)

    # Get a new task from curriculum, with retry logic for invalid configurations
    max_retries = 10
    for attempt in range(max_retries):
        try:
            self._current_task = self._curriculum.get_task()
            self._env.set_mg_config(self._current_task.get_env_cfg())
            break
        except (AssertionError, ValueError) as e:
            if attempt < max_retries - 1:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Task configuration error (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Resampling new task..."
                )
                if hasattr(self._current_task, "_task_id"):
                    self._current_task.complete(-1.0)
                continue
            else:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(
                    f"Failed to find valid task configuration after {max_retries} attempts. "
                    f"Last error: {e}"
                )
                raise

    return obs, info
```

**Simplify `step()` method**:
```python
def step(self, *args, **kwargs):
    """Step the environment and handle task completion."""
    obs, rewards, terminals, truncations, infos = self._env.step(*args, **kwargs)

    if terminals.all() or truncations.all():
        mean_reward = self._env.get_episode_rewards().mean()
        self._current_task.complete(mean_reward)
        self._curriculum.update_task_performance(self._current_task._task_id, mean_reward)

        # ALWAYS emit per-label sample count (needed for basic curriculum monitoring)
        label = self._current_task.get_label()
        if label is not None and isinstance(label, str):
            if "curriculum_stats/per_label_samples_this_epoch" not in infos:
                infos["curriculum_stats/per_label_samples_this_epoch"] = {}
            infos["curriculum_stats/per_label_samples_this_epoch"][label] = 1

        # Track task completions for troubleshooting (ONLY if flag enabled)
        if self._enable_per_label_tracking:
            task_id = self._current_task._task_id
            if task_id not in self._tracked_task_ids and len(self._tracked_task_ids) < 3:
                self._tracked_task_ids.append(task_id)

            if task_id in self._tracked_task_ids:
                self._tracked_task_completions_this_epoch[task_id] = (
                    self._tracked_task_completions_this_epoch.get(task_id, 0) + 1
                )

        # Get new task with retry logic
        max_retries = 10
        for attempt in range(max_retries):
            try:
                self._current_task = self._curriculum.get_task()
                self._env.set_mg_config(self._current_task.get_env_cfg())
                break
            except (AssertionError, ValueError) as e:
                if attempt < max_retries - 1:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"Task configuration error on episode completion "
                        f"(attempt {attempt + 1}/{max_retries}): {e}. Resampling..."
                    )
                    if hasattr(self._current_task, "_task_id"):
                        self._current_task.complete(-1.0)
                    continue
                else:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(
                        f"Failed to find valid task configuration after {max_retries} attempts "
                        f"on episode completion. Last error: {e}"
                    )
                    raise

    return obs, rewards, terminals, truncations, infos
```

**Simplify `__getattribute__` method**:
```python
def __getattribute__(self, name: str):
    """Intercept all attribute access and delegate to wrapped environment when appropriate."""
    # Only handle our own attributes
    if name in (
        "_env",
        "_curriculum",
        "_current_task",
        "_enable_per_label_tracking",
        "_tracked_task_ids",
        "_tracked_task_completions_this_epoch",
        "_tracked_task_completions_baseline",
        "step",
        "reset",
    ):
        return object.__getattribute__(self, name)

    # Delegate to wrapped environment
    try:
        env = object.__getattribute__(self, "_env")
        return getattr(env, name)
    except AttributeError:
        return object.__getattribute__(self, name)
```

**Update docstring**:
```python
class CurriculumEnv(PufferEnv):
    """Environment wrapper that integrates with a curriculum system.

    This wrapper:
    - Handles task selection and completion through the curriculum
    - ALWAYS emits per-label episode counts (needed for basic curriculum monitoring)
    - Optionally tracks first 3 tasks for debugging (if show_curriculum_troubleshooting_logging=True)
    - All other curriculum stats are collected centrally at epoch boundaries by StatsReporter
    """
```

---

### Step 3: Remove Epoch Reset Calls

**File**: `experiments/recipes/curriculum_test/task_dependency_simulator.py`

**Delete**:
```python
# DELETE this loop:
for env in envs:
    env.reset_epoch_counters()
```

The `StatsReporter` already clears `rollout_stats` after each epoch in its `clear_rollout_stats()` method, so no explicit reset is needed.

---

### Step 4: Update Per-Label Stats Processing

**File**: `metta/rl/stats.py`

**Keep existing logic** - it already handles dict-valued stats correctly:
```python
# This already works for per_label_samples_this_epoch
if "per_label_samples_this_epoch" in k or "tracked_task_completions_this_epoch" in k:
    mean_stats[k] = np.sum(v)
```

No changes needed here.

---

### Step 5: Add Per-Label Tracking to Curriculum Algorithm (REQUIRED)

**File**: `metta/cogworks/curriculum/learning_progress_algorithm.py`

**NOTE**: This step is REQUIRED (not optional) because Step 1 depends on this method.

```python
def get_per_label_lp_scores(self) -> dict[str, dict[str, float]]:
    """Get per-label LP scores for troubleshooting.

    Returns:
        Dict mapping label -> {raw, postzscored, prob}
    """
    if not self.hypers.show_curriculum_troubleshooting_logging:
        return {}

    per_label = {}
    for task_id, task in self._task_pool.items():
        label = task.get_label()
        if not label or not isinstance(label, str):
            continue

        if label not in per_label:
            per_label[label] = {"raw": 0.0, "postzscored": 0.0, "prob": 0.0, "count": 0}

        # Accumulate scores (will average later)
        per_label[label]["raw"] += self.get_task_raw_lp_score(task_id)
        per_label[label]["postzscored"] += self.get_task_postzscored_lp_score(task_id)
        per_label[label]["prob"] += self.get_task_lp_score(task_id)
        per_label[label]["count"] += 1

    # Average scores per label
    for label, scores in per_label.items():
        count = scores.pop("count")
        if count > 0:
            scores["raw"] /= count
            scores["postzscored"] /= count
            scores["prob"] /= count

    return per_label
```

---

## Code Removal Summary

### From CurriculumEnv

**Deleted Attributes** (8 items):
- `_stats_update_counter`
- `_stats_update_frequency`
- `_cached_curriculum_stats`
- `_curriculum_stats_cache_valid`
- `_per_label_lp_scores`
- `_per_label_postzscored_lp_scores`
- `_per_label_lp_probs`
- `_task_slot_input_rewards`

**Kept Attributes** (gated by troubleshooting flag):
- `_tracked_task_ids` - Only allocated if show_curriculum_troubleshooting_logging=True
- `_tracked_task_completions_this_epoch` - Only allocated if flag enabled
- `_tracked_task_completions_baseline` - Only allocated if flag enabled

**Deleted Methods** (5 items):
- `reset_epoch_counters()`
- `_add_curriculum_stats_to_info()`
- `_calculate_gini_coefficient()`
- `set_stats_update_frequency()`
- `force_stats_update()`

**Simplified Methods**:
- `reset()` - Remove cache invalidation, tracked task logic
- `step()` - Remove all stat tracking except per_label_samples
- `__getattribute__` - Remove references to deleted attributes

**Lines of Code Removed**: ~200 lines

**Key Changes from Initial Plan**:
- `per_label_samples_this_epoch` now ALWAYS emitted (not gated) - needed for basic monitoring
- Tracked task attributes (`_tracked_task_ids`, etc.) kept but gated by flag for performance
- Step 5 changed from "Optional" to "REQUIRED" - Step 1 depends on it

### From task_dependency_simulator.py

**Deleted**:
- `env.reset_epoch_counters()` call in simulation loop

---

## Benefits

### Performance
- ✅ **50x reduction in stat updates**: From every 50 steps per env → once per epoch
- ✅ **Eliminates caching overhead**: No more cache invalidation checks
- ✅ **Reduces info dict size**: Only 1 stat instead of 15+

### Quality
- ✅ **Smooth WandB graphs**: Consistent timing, no choppiness
- ✅ **Accurate values**: No more averaging shared global stats across envs
- ✅ **Clear semantics**: Epoch-level stats logged at epoch level

### Maintainability
- ✅ **200 fewer lines** in CurriculumEnv
- ✅ **Clear separation**: Per-episode counts vs global stats
- ✅ **Easier debugging**: Single collection point

---

## Testing Checklist

- [ ] Run navigation training with curriculum
- [ ] Verify all curriculum stats appear in WandB
- [ ] Check per_label_samples_this_epoch sums correctly
- [ ] Verify troubleshooting stats only appear when flag enabled
- [ ] Compare values before/after migration (should match)
- [ ] Check graphs are smooth (not choppy)
- [ ] Run task_dependency_simulator tests
- [ ] Verify no performance regression

---

## Rollout Plan

1. **Implement StatsReporter changes** (Step 1)
2. **Test with existing logging still active**
3. **Simplify CurriculumEnv** (Step 2)
4. **Clean up simulator** (Step 3)
5. **Run full test suite**
6. **Deploy to staging**
7. **Monitor production run**

**Estimated effort**: 2-3 hours implementation + 1 hour testing

---

## Notes

- The `per_label_samples_this_epoch` stat is the only one that truly needs per-episode granularity
- All other curriculum stats are global/aggregate and should be sampled once per epoch
- The Gini coefficient calculation is kept in both places (not consolidated per user request)
- Troubleshooting stats (Groups C & D) are now consistently gated by the same flag
- Epoch reset is no longer needed because StatsReporter already clears rollout_stats
- `per_label_samples_this_epoch` is ALWAYS emitted (not behind troubleshooting flag) for basic monitoring
- Tracked task dynamics (first 3 tasks) moved behind troubleshooting flag to reduce overhead

