# 🐛 Bug Report: Evaluation system breaks sweep functionality

## Summary
The trainer evaluation system has two critical bugs that break sweep functionality:
1. **Hardcoded "reward" metric**: Ignores sweep config `metric` field, preventing optimization of other metrics
2. **Missing top-level score**: No `metadata["score"]` written to policies, breaking policy selection

## Problem Details

### Bug #1: Hardcoded "reward" Metric
**Current Broken Flow**:
1. **Sweep config specifies**: `metric: episode_length` (or any non-reward metric)
2. **Trainer ignores sweep config**: Always queries `"reward"` from database (line 762 in `metta/rl/trainer.py`)
3. **Results stored as**: `{category}/score` (misleading - actually contains reward data)
4. **Policy selection**: Uses reward-based scoring regardless of intended optimization target
5. **Protein optimization**: Tries to optimize wrong metric

### Bug #2: Missing Top-Level Score (CRITICAL)
**Current Broken Flow**:
1. **Evaluation runs**: Populates `self.evals` with category scores
2. **No score assignment**: **Missing** `metadata["score"]` write completely
3. **Policy selection fails**: `PolicyStore.get_best_policy(metric="score")` fails or picks randomly
4. **Sweep gets wrong policy**: No principled checkpoint selection within runs

### Code Location
**File**: `metta/rl/trainer.py`
**Lines**: 761-775
**Current problematic code**:
```python
# Bug #1: Hardcoded "reward"
for category in categories:
    score = stats_db.get_average_metric_by_filter(
        "reward", self.latest_saved_policy_record, f"sim_name LIKE '%{category}%'"  # ❌ HARDCODED
    )
    # ...
    self.evals[f"{category}/score"] = score

# Bug #2: Missing metadata assignment
# ... evaluation ends here, NO metadata["score"] written! ❌
```

## Expected Behavior
The trainer should:
1. Read the target metric from sweep configuration (`cfg.sweep.metric`)
2. Query that specific metric from the stats database
3. Store results using the actual metric name (not hardcoded "score")
4. **Write aggregate score to policy metadata for selection**
5. Enable optimization of any available metric

## Proposed Fix

### Complete Solution (fixes both bugs)
```python
# Get target metric from sweep config
target_metric = getattr(cfg, 'sweep', {}).get('metric', 'reward')  # fallback to reward

# Collect category scores using correct metric
for category in categories:
    score = stats_db.get_average_metric_by_filter(
        target_metric, self.latest_saved_policy_record, f"sim_name LIKE '%{category}%'"
    )
    if score is not None:
        self.evals[f"{category}/{target_metric}"] = score

# CRITICAL: Write top-level score for policy selection
category_scores = [v for k, v in self.evals.items() if k.endswith(f"/{target_metric}")]
if category_scores and self.latest_saved_policy_record:
    self.latest_saved_policy_record.metadata["score"] = float(np.mean(category_scores))

# Store detailed scores and save policy
if self.latest_saved_policy_record and self.evals:
    if "eval_scores" not in self.latest_saved_policy_record.metadata:
        self.latest_saved_policy_record.metadata["eval_scores"] = {}
    self.latest_saved_policy_record.metadata["eval_scores"].update(self.evals)
    self.policy_store.save(self.latest_saved_policy_record)
```

### Minimal Fix (addresses missing score only)
```python
# After existing evaluation code, add:
scores = [v for k, v in self.evals.items() if k.endswith("/score")]
if scores and self.latest_saved_policy_record:
    self.latest_saved_policy_record.metadata["score"] = float(np.mean(scores))
```

## Impact Assessment

### Bug #1 (Hardcoded Reward)
- **High**: Breaks any sweep not optimizing for reward
- **Scope**: All sweep experiments using non-reward optimization targets
- **Silent failure**: No error thrown, just optimizes wrong metric

### Bug #2 (Missing Score) - **CRITICAL**
- **Critical**: Breaks ALL sweep policy selection
- **Scope**: Every sweep run - policy selection is essentially random
- **Silent failure**: No error, but sweep picks suboptimal policies
- **Performance impact**: Massive - sweeps can't find good hyperparameters

## Reproducibility

### Bug #1:
1. Create sweep config with `metric: episode_length`
2. Run sweep
3. Observe that policy selection still uses reward data

### Bug #2:
1. Run any sweep: `./devops/sweep.sh run=test.sweep`
2. Check policy metadata: `metadata["score"]` is missing
3. `PolicyStore.get_best_policy(metric="score")` fails or picks randomly
4. Sweep evaluation gets wrong checkpoint

## Related Issues
- Misleading storage format (`/score` suffix regardless of actual metric)
- Mismatch between sweep config expectations and trainer behavior
- Lack of configuration validation for supported metrics
- **Policy selection randomness masking hyperparameter optimization effectiveness**

---
**Priority**: **CRITICAL** - Bug #2 breaks all sweep policy selection, Bug #1 breaks non-reward optimization
