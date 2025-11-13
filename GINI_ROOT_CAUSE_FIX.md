# Gini Score Root Cause Analysis and Fix

## Problem

All Gini scores except `pool_composition_gini` were returning zero despite clear sampling dynamics. Specifically:

- ✅ `pool_composition_gini` = 0.299 (working)
- ❌ `sampling_gini` = 0.000 (broken, now fixed)
- ❌ `eviction_gini` = missing (broken, now fixed)
- ❌ `per_epoch_samples_gini` = missing (broken, now fixed)

## Investigation Process

### Solution 1: Flag Dependency (Implemented)

Initially suspected the `show_curriculum_troubleshooting_logging` flag was preventing per-label LP stats collection
needed for `sampling_gini`.

**Fix**: Made per-label LP score collection unconditional in `stats_reporter.py` lines 659-663.

### Solution 2: Diagnostic Logging (Implemented)

Added comprehensive logging to trace data flow through the Gini calculation pipeline.

**Key Discovery from Logs**:

```
[17:43:29.503989] INFO     Rollout stats keys (sample): []
```

The `rollout_stats` dictionary was **completely empty** when trying to calculate Gini scores!

### Root Cause Discovery

By tracing the data flow, we discovered:

1. **`CurriculumEnv` emits stats correctly** (`curriculum_env.py` lines 106-128):
   - `per_label_evictions_this_epoch` as nested dict
   - `per_label_samples_this_epoch` as nested dict

2. **Stats are accumulated via `process_rollout()`** which calls `accumulate_rollout_stats()`:
   - Expects a list of info dicts
   - Uses `unroll_nested_dict()` to flatten them
   - Accumulates into `self._state.rollout_stats`

3. **THE BUG**: In `metta/rl/training/core.py` line 168:

   ```python
   infos_list: list[dict[str, Any]] = list(info) if info else []
   ```

   When `info` is a **dict** (which it is from `CurriculumEnv`), `list(info)` returns a **list of keys**, not a list
   containing the dict!

   Example:

   ```python
   info = {"env_curriculum_stats/per_label_samples_this_epoch": {"label1": 5}}
   list(info)  # Returns ["env_curriculum_stats/per_label_samples_this_epoch"]
   # NOT [{"env_curriculum_stats/per_label_samples_this_epoch": {"label1": 5}}]
   ```

   Result: Curriculum stats were **never being accumulated** into `rollout_stats`, causing it to remain empty.

## The Fix

### File: `metta/rl/training/core.py` (lines 168-177)

**Before**:

```python
infos_list: list[dict[str, Any]] = list(info) if info else []
if infos_list:
    raw_infos.extend(infos_list)
```

**After**:

```python
# Handle both dict and list info formats
# If info is a dict, wrap it in a list; if it's already a list, use as-is
infos_list: list[dict[str, Any]] = []
if info:
    if isinstance(info, dict):
        infos_list = [info]
    elif isinstance(info, list):
        infos_list = info
if infos_list:
    raw_infos.extend(infos_list)
```

## Supporting Changes

### File: `metta/rl/training/stats_reporter.py`

1. **Made per-label LP collection unconditional** (lines 659-663):
   - Required for `sampling_gini` calculation
   - No longer gated by `show_curriculum_troubleshooting_logging`

2. **Added diagnostic logging** throughout Gini pipeline:
   - Logs data availability before calculations
   - Warns on missing data
   - Shows sample values for debugging

3. **Added `_reconstruct_dict_from_flattened_keys()` helper** (lines 821-835):
   - Reconstructs nested dicts from flattened rollout_stats keys
   - Used by `eviction_gini` and `per_epoch_samples_gini`

4. **Reordered execution** (lines 659-670):
   - Per-label LP stats collected before Gini calculations
   - Ensures data dependencies are met

## Expected Results

After this fix, all Gini scores should work correctly:

- ✅ **`pool_composition_gini`**: Inequality in task pool composition by label
- ✅ **`sampling_gini`**: Inequality in LP-based sampling probabilities by label
- ✅ **`eviction_gini`**: Inequality in which labels are evicted this epoch
- ✅ **`per_epoch_samples_gini`**: Inequality in episode completions by label this epoch
- ✅ **`task_sampling_gini`**: Inequality in individual task sampling (pass-through from algorithm)
- ✅ **`task_lp_gini`**: Inequality in individual task LP scores (pass-through from algorithm)

## Data Flow Summary

```
CurriculumEnv.step()
  └─> Emits info dict with nested curriculum stats
      └─> core.py rollout_phase()  [FIXED HERE]
          └─> Wraps dict in list: [info]
              └─> stats_reporter.accumulate_infos()
                  └─> accumulate_rollout_stats()
                      └─> Flattens with unroll_nested_dict()
                          └─> Accumulates into rollout_stats
                              └─> stats_reporter._collect_curriculum_stats()
                                  └─> _reconstruct_dict_from_flattened_keys()
                                      └─> Calculate Gini coefficients
```

## Testing

To verify the fix works:

1. Run training with curriculum
2. Check logs for non-empty rollout_stats:
   ```
   Rollout stats keys (sample): ['env_curriculum_stats/per_label_samples_this_epoch/label1', ...]
   ```
3. Verify all Gini scores are non-zero (when appropriate)
4. Check WandB for `curriculum_stats/*_gini` metrics

## Related Files Modified

- `metta/rl/training/core.py` - Root cause fix
- `metta/rl/training/stats_reporter.py` - Supporting changes and diagnostics
- `metta/rl/stats.py` - Already had sum vs mean fix from previous PR
