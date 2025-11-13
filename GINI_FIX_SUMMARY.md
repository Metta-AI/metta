# Gini Coefficient Calculation Fix

## Problem Summary

All Gini coefficients were returning zero despite clear sampling dynamics in `variant_maps` training. Investigation revealed two interconnected bugs in the stats collection pipeline.

## Root Causes

### Bug 1: Execution Order (Timing Bug)
**Location:** `metta/rl/training/stats_reporter.py:_collect_curriculum_stats()`

Gini calculations were running **before** the data they needed was collected:

```python
# OLD ORDER (BROKEN):
Line 659-711: Calculate Gini coefficients
  - sampling_gini (line 682) looks for curriculum_stats/per_label_lp_probs/*
  - eviction_gini (line 695) looks for dict at env_curriculum_stats/per_label_evictions_this_epoch
  - per_epoch_samples_gini (line 707) looks for dict at env_curriculum_stats/per_label_samples_this_epoch

Line 771-774: Collect troubleshooting stats
  - Adds per_label_lp_probs/* keys (line 882 in helper)
```

The `per_label_lp_probs/*` keys were added by troubleshooting stats collection **after** the Gini calculation tried to use them, resulting in empty lists and zero Gini values.

### Bug 2: Dictionary Flattening (Lookup Bug)
**Location:** `metta/rl/stats.py:accumulate_rollout_stats()` + stats_reporter.py

Stats were emitted as nested dicts but flattened before Gini calculations could access them:

1. **Emission** (curriculum_env.py:124-128):
   ```python
   infos["env_curriculum_stats/per_label_samples_this_epoch"] = {
       "small_extractor_hub_30_lonely_heart": 2,
       "small_extractor_hub_30_pack_rat": 3
   }
   ```

2. **Flattening** (stats.py:36 via `unroll_nested_dict()`):
   ```python
   "env_curriculum_stats/per_label_samples_this_epoch/small_extractor_hub_30_lonely_heart": 2
   "env_curriculum_stats/per_label_samples_this_epoch/small_extractor_hub_30_pack_rat": 3
   ```

3. **Failed Lookup** (stats_reporter.py:703):
   ```python
   per_label_samples_key = "env_curriculum_stats/per_label_samples_this_epoch"
   if per_label_samples_key in self._state.rollout_stats:  # ❌ NEVER TRUE
       # Parent key no longer exists after flattening!
   ```

## Solution Implemented

### Fix 1: Reorder Execution
Moved troubleshooting stats collection to **before** Gini calculations:

```python
# NEW ORDER (FIXED):
Line 659-664: Collect troubleshooting stats FIRST
  - Populates per_label_lp_probs/* keys

Line 666-719: Calculate Gini coefficients
  - Now all required data exists in stats dict
```

### Fix 2: Reconstruct Dictionaries
Added helper function to reconstruct dicts from flattened keys:

```python
@staticmethod
def _reconstruct_dict_from_flattened_keys(
    rollout_stats: dict,
    prefix: str
) -> dict[str, float]:
    """Reconstruct dictionary from flattened keys.

    Converts:
        "prefix/label1": [2, 1, 3]
        "prefix/label2": [3, 2]

    To:
        {"label1": 6, "label2": 5}
    """
    result = {}
    prefix_with_slash = f"{prefix}/"
    for key, value in rollout_stats.items():
        if key.startswith(prefix_with_slash):
            label = key[len(prefix_with_slash):]
            if isinstance(value, list):
                result[label] = float(sum(value))  # Sum for count-based stats
            else:
                result[label] = float(value)
    return result
```

Updated Gini calculations to use helper:
```python
# Eviction gini
per_label_evictions = self._reconstruct_dict_from_flattened_keys(
    self._state.rollout_stats,
    "env_curriculum_stats/per_label_evictions_this_epoch"
)

# Sample gini
per_label_samples = self._reconstruct_dict_from_flattened_keys(
    self._state.rollout_stats,
    "env_curriculum_stats/per_label_samples_this_epoch"
)
```

## Files Changed

1. **metta/rl/training/stats_reporter.py**
   - Moved troubleshooting stats collection before Gini calculations
   - Added `_reconstruct_dict_from_flattened_keys()` helper method
   - Updated eviction_gini and per_epoch_samples_gini to use helper
   - Removed duplicate troubleshooting stats collection

## Affected Gini Metrics (Now Fixed)

| Metric | Old Status | Issue | New Status |
|--------|-----------|-------|------------|
| `curriculum_stats/sampling_gini` | ❌ Zero | Timing bug | ✅ Fixed |
| `curriculum_stats/eviction_gini` | ❌ Zero | Flattening bug | ✅ Fixed |
| `curriculum_stats/per_epoch_samples_gini` | ❌ Zero | Flattening bug | ✅ Fixed |
| `curriculum_stats/pool_composition_gini` | ✅ Working | No issue | ✅ Still works |
| `curriculum_stats/task_sampling_gini` | ✅ Working | No issue | ✅ Still works |

## Testing

All existing tests pass:
```bash
uv run pytest tests/rl/test_dict_stats_aggregation.py -v
# 3 passed
```

## Expected Behavior After Fix

When running `variant_maps.train`:
- ✅ `sampling_gini` should show non-zero values reflecting LP-based sampling inequality
- ✅ `eviction_gini` should show non-zero values when evictions occur
- ✅ `per_epoch_samples_gini` should show non-zero values reflecting sampling distribution
- ✅ Gini values should change dynamically as the curriculum adapts
- ✅ Logs should show: "Sampling gini: 0.XXX (N labels, probs=[...])"

## Why This Matters for variant_maps

The `variant_maps` recipe is particularly affected because:
- **42-72 unique labels** (many variant combinations)
- **Learning progress-based sampling** creates natural inequality
- **Zero Gini incorrectly suggests uniform sampling** when LP is actively differentiating

With the fix, Gini coefficients now accurately reflect curriculum dynamics, enabling proper monitoring of learning progress-driven task selection.

