# Comprehensive Gini Coefficient Diagnostics

## Overview

The curriculum system now calculates Gini coefficients at **8 key stages** of the LP calculation and task management pipeline. This helps diagnose exactly where selectivity is being lost or concentrated.

## What is a Gini Coefficient?

The Gini coefficient measures inequality in a distribution:
- **0.0** = Perfect equality (all values are equal)
- **1.0** = Perfect inequality (one value has everything)

For curriculum learning:
- **Higher Gini** = More selective (focused on specific tasks/labels)
- **Lower Gini** = More uniform (spread across many tasks/labels)

## The 8 Gini Metrics

### 1. `gini/pool_occupancy`
**What it measures**: Inequality in how often tasks are sampled (completion counts)

**Interpretation**:
- High (>0.5): Some tasks are used much more than others (good for exploitation)
- Low (<0.3): Tasks are sampled uniformly (good for exploration)
- **Expected behavior**: Should increase during training as curriculum focuses on high-LP tasks

**WandB key**: `curriculum_gini/pool_occupancy`

---

### 2. `gini/raw_lp_scores`
**What it measures**: Inequality in raw learning progress scores (before any transformation)

**Interpretation**:
- High (>0.5): Large variation in LP signals - some tasks have much higher LP than others
- Low (<0.3): Most tasks have similar LP scores - curriculum has difficulty differentiating
- **This is the "source" signal** - if low here, transformations can't add selectivity

**WandB key**: `curriculum_gini/raw_lp_scores`

**Critical**: This should generally be the **highest Gini** in the pipeline. If it's low, your LP calculation isn't differentiating tasks well.

---

### 3. `gini/raw_lp_by_label`
**What it measures**: Inequality in raw LP aggregated by task labels

**Interpretation**:
- High (>0.5): Some task types have much higher cumulative LP than others
- Low (<0.3): All task types have similar total LP
- Shows whether LP signal varies more **within labels** or **between labels**

**WandB key**: `curriculum_gini/raw_lp_by_label`

**Use case**: If this is much lower than `gini/raw_lp_scores`, LP varies mostly within labels (good). If similar, LP varies mostly between labels.

---

### 4. `gini/zscored_lp_scores`
**What it measures**: Inequality in z-score normalized LP (after centering/scaling, before sigmoid)

**Interpretation**:
- Shows the effect of z-score normalization on selectivity
- Z-score normalization (temperature=0) tends to **reduce** Gini by standardizing the range
- High values here mean strong relative differences survived normalization

**WandB key**: `curriculum_gini/zscored_lp_scores`

**Expected**: Should be **lower** than `gini/raw_lp_scores` due to normalization

---

### 5. `gini/sampling_probs`
**What it measures**: Inequality in final sampling probabilities (after sigmoid + normalization)

**Interpretation**:
- High (>0.5): Sampling is highly concentrated on a few tasks
- Low (<0.3): Sampling is spread uniformly across tasks
- **This is the "output" of the LP pipeline** - what actually drives sampling

**WandB key**: `curriculum_gini/sampling_probs`

**Expected**: Should be the **lowest** Gini because:
1. Sigmoid smooths extreme values
2. Normalization to sum=1 reduces inequality
3. Exploration bonus adds probability floor

---

### 6. `gini/sampling_by_label`
**What it measures**: Inequality in actual sampling counts aggregated by label

**Interpretation**:
- High (>0.5): Training is focused on specific task types
- Low (<0.3): Training is spread evenly across task types
- **This is empirical** - measures what actually happened, not just probabilities

**WandB key**: `curriculum_gini/sampling_by_label`

**Critical**: Compare to `gini/raw_lp_by_label` to see if LP signal translates to actual sampling

---

### 7. `gini/evictions_by_label`
**What it measures**: Inequality in eviction counts aggregated by label

**Interpretation**:
- High (>0.5): Some task types are evicted much more than others (low LP)
- Low (<0.3): Evictions are spread evenly across task types
- Shows which task types are being deprioritized

**WandB key**: `curriculum_gini/evictions_by_label`

**Use case**: High eviction Gini + high sampling Gini for same label = curriculum is strongly differentiating that task type

---

### 8. `gini/pool_composition_by_label`
**What it measures**: Inequality in how many tasks of each label exist in the pool

**Interpretation**:
- High (>0.5): Pool is dominated by specific task types
- Low (<0.3): Pool has equal representation of task types
- Shows task generator + eviction balance

**WandB key**: `curriculum_gini/pool_composition_by_label`

**Expected**: Should be relatively **low** if task generator is balanced and eviction is LP-based

---

## Selectivity Loss Metrics

### `selectivity_loss/lp_to_prob`
**Calculation**: `gini/raw_lp_scores - gini/sampling_probs`

**What it measures**: How much inequality is lost between raw LP signal and final sampling

**Interpretation**:
- **Positive** (typical): Transformations reduce inequality (sigmoid, normalization, exploration)
- **Large positive** (>0.3): Pipeline is smoothing too aggressively
- **Negative** (unusual): Final distribution is more selective than raw LP (check for bugs)

**Ideal range**: 0.1 - 0.3

---

### `selectivity_loss/lp_label_to_sampling_label`
**Calculation**: `gini/raw_lp_by_label - gini/sampling_by_label`

**What it measures**: How well label-level LP translates to label-level sampling

**Interpretation**:
- **Near zero**: LP signal accurately drives label sampling
- **Large positive**: Sampling is more uniform than LP suggests
- **Negative**: Sampling is more selective than LP (possibly due to pool composition)

**Ideal**: Should be small (<0.2)

---

## Diagnostic Flow Chart

```
1. Check gini/raw_lp_scores:
   Low (<0.3)? → LP calculation isn't differentiating tasks
                  ↓
                  - Check task performance variance
                  - Check EMA parameters (timescales)
                  - Verify sufficient samples per task

   High (>0.5)? → LP signal is strong, check if it propagates
                  ↓
                  Continue to step 2

2. Compare gini/raw_lp_scores to gini/sampling_probs:
   Loss > 0.4? → Pipeline is over-smoothing
                 ↓
                 - Reduce progress_smoothing
                 - Adjust temperature/z-score
                 - Lower exploration_bonus

   Loss < 0.2? → Good propagation of LP signal
                 ↓
                 Continue to step 3

3. Compare gini/sampling_probs to gini/sampling_by_label:
   Task-level >> Label-level? → Variation is within labels (good)
   Label-level >> Task-level? → Variation is between labels only

4. Check gini/pool_composition_by_label:
   High (>0.5)? → Pool is unbalanced
                  ↓
                  - Check task generator balance
                  - Review eviction policy
```

## Example Analysis

### Scenario 1: "Lost Selectivity in Pipeline"
```
gini/raw_lp_scores:        0.52  (strong signal)
gini/zscored_lp_scores:    0.31  (normalized)
gini/sampling_probs:       0.18  (weak output)
selectivity_loss/lp_to_prob: 0.34  (high loss)
```

**Diagnosis**: LP signal is strong but gets heavily smoothed. The z-score normalization and sigmoid are reducing selectivity too much.

**Solution**:
- Use temperature scaling (lp_score_temperature > 0) instead of z-score
- Reduce progress_smoothing
- Lower exploration_bonus

---

### Scenario 2: "Weak Source Signal"
```
gini/raw_lp_scores:        0.15  (weak signal)
gini/sampling_probs:       0.12  (weak output)
selectivity_loss/lp_to_prob: 0.03  (low loss)
```

**Diagnosis**: Pipeline is propagating signal well, but the LP calculation itself isn't differentiating tasks.

**Solution**:
- Check task performance variance
- Increase EMA timescales for faster response
- Ensure tasks have sufficient samples
- Verify baseline normalization is working

---

### Scenario 3: "Label-Level Uniformity Despite Task-Level Selectivity"
```
gini/raw_lp_scores:        0.48  (task-level selectivity)
gini/raw_lp_by_label:      0.12  (label-level uniformity)
gini/sampling_by_label:    0.14  (empirical uniformity)
```

**Diagnosis**: High LP variation within each label, but labels have similar total LP. This is actually **good** - curriculum is selecting the best tasks from each task type rather than biasing toward one type.

**No action needed** unless you want to focus on specific task types.

---

## Monitoring in WandB

All Gini metrics are logged under the `algorithm/` prefix:

```
curriculum_gini/pool_occupancy
curriculum_gini/raw_lp_scores
curriculum_gini/raw_lp_by_label
curriculum_gini/zscored_lp_scores
curriculum_gini/sampling_probs
curriculum_gini/sampling_by_label
curriculum_gini/evictions_by_label
curriculum_gini/pool_composition_by_label
curriculum_gini/selectivity_loss_lp_to_prob
curriculum_gini/selectivity_loss_lp_label_to_sampling_label
```

### Recommended WandB Charts

1. **Gini Pipeline Chart**: Line plot with all 5 main Ginis:
   - raw_lp_scores
   - zscored_lp_scores
   - sampling_probs
   - sampling_by_label
   - pool_composition_by_label

2. **Selectivity Loss Chart**: Line plot of both loss metrics

3. **Label Analysis Chart**: Compare:
   - raw_lp_by_label
   - sampling_by_label
   - evictions_by_label
   - pool_composition_by_label

## Configuration Parameters That Affect Gini

### Increase Selectivity (Higher Gini)
- **Lower** `progress_smoothing` (default: 0.0) - reduces probability floor
- **Lower** `exploration_bonus` (default: 0.1) - reduces boost to unexplored tasks
- **Higher** `lp_score_temperature` (if > 0) - amplifies LP differences
- **Higher** `eviction_threshold_percentile` - evict more low-LP tasks

### Decrease Selectivity (Lower Gini, More Uniform)
- **Higher** `progress_smoothing` - adds probability floor to all tasks
- **Higher** `exploration_bonus` - boosts unexplored tasks more
- **Lower** `lp_score_temperature` (if > 0) - smooths LP differences
- Use **z-score normalization** (`lp_score_temperature=0`) - standardizes range

## Performance Considerations

Calculating 8 Gini coefficients is more expensive than the previous single calculation. The comprehensive Gini calculation:

- Iterates through all tasks: **O(N)** where N = num_active_tasks
- Calls scorer methods for each task: **O(N)**
- Calculates 8 Gini coefficients: **O(N log N)** each (due to sorting)
- **Total**: **O(N log N)**

For typical pool sizes (N=1000), this adds ~5-10ms per stats collection.

**Recommendation**: These stats are in `get_base_stats()`, which is called every training iteration. If performance becomes an issue, consider:
1. Moving some Ginis to `get_detailed_stats()` (called less frequently)
2. Caching Gini calculations
3. Reducing pool size

## Summary

The comprehensive Gini tracking lets you:

1. **Identify where selectivity is lost** in the LP → sampling pipeline
2. **Validate LP signal strength** at the source
3. **Monitor label-level vs task-level** variation
4. **Track empirical sampling behavior** against intended probabilities
5. **Detect pipeline bugs** (e.g., negative selectivity loss)
6. **Tune hyperparameters** to achieve desired selectivity

This diagnostic framework makes curriculum behavior transparent and debuggable.

