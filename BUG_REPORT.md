# 🐛 Bug Report: Hardcoded "reward" metric breaks sweep flexibility

## Summary
The trainer evaluation logic hardcodes `"reward"` as the metric to query from the stats database, ignoring the `metric` field specified in sweep configurations. This prevents sweeps from optimizing for other metrics like episode length, success rate, or custom evaluation criteria.

## Problem Details

### Current Broken Flow
1. **Sweep config specifies**: `metric: episode_length` (or any non-reward metric)
2. **Trainer ignores sweep config**: Always queries `"reward"` from database (line 762 in `metta/rl/trainer.py`)
3. **Results stored as**: `{category}/score` (misleading - actually contains reward data)
4. **Policy selection**: Uses reward-based scoring regardless of intended optimization target
5. **Protein optimization**: Tries to optimize wrong metric

### Code Location
**File**: `metta/rl/trainer.py`
**Lines**: 761-763
**Problematic code**:
```python
score = stats_db.get_average_metric_by_filter(
    "reward", self.latest_saved_policy_record, f"sim_name LIKE '%{category}%'"  # ❌ HARDCODED
)
```

## Expected Behavior
The trainer should:
1. Read the target metric from sweep configuration (`cfg.sweep.metric`)
2. Query that specific metric from the stats database
3. Store results using the actual metric name (not hardcoded "score")
4. Enable optimization of any available metric

## Proposed Fix

### Option 1: Dynamic metric query
```python
# Get target metric from sweep config
target_metric = getattr(cfg, 'sweep', {}).get('metric', 'reward')  # fallback to reward

for category in categories:
    score = stats_db.get_average_metric_by_filter(
        target_metric, self.latest_saved_policy_record, f"sim_name LIKE '%{category}%'"
    )
    if score is not None:
        self.evals[f"{category}/{target_metric}"] = score

# Compute aggregate for policy selection
category_scores = [v for k, v in self.evals.items() if k.endswith(f"/{target_metric}")]
if category_scores:
    self.latest_saved_policy_record.metadata["score"] = float(np.mean(category_scores))
```

### Option 2: Multi-metric support
```python
# Support multiple metrics, with primary for policy ranking
metrics_to_collect = [cfg.sweep.metric] + ['reward', 'episode_length']  # collect common ones

for metric in metrics_to_collect:
    for category in categories:
        score = stats_db.get_average_metric_by_filter(
            metric, self.latest_saved_policy_record, f"sim_name LIKE '%{category}%'"
        )
        if score is not None:
            self.evals[f"{category}/{metric}"] = score

# Use primary metric for policy ranking
primary_scores = [v for k, v in self.evals.items() if k.endswith(f"/{cfg.sweep.metric}")]
if primary_scores:
    self.latest_saved_policy_record.metadata["score"] = float(np.mean(primary_scores))
```

## Impact
- **High**: Breaks any sweep not optimizing for reward
- **Scope**: All sweep experiments using non-reward optimization targets
- **Silent failure**: No error thrown, just optimizes wrong metric

## Reproducibility
1. Create sweep config with `metric: episode_length`
2. Run sweep
3. Observe that policy selection still uses reward data
4. Protein optimizes for reward instead of episode length

## Related Issues
- Misleading storage format (`/score` suffix regardless of actual metric)
- Mismatch between sweep config expectations and trainer behavior
- Lack of configuration validation for supported metrics

---
**Priority**: High - breaks core sweep functionality for non-reward optimization
