# LP-Based Task Generation Sampling

## Overview

This document describes a future enhancement to scale the sampling of newly generated tasks by their label's Learning Progress (LP) scores, with configurable minimum sampling probabilities that can adapt during training.

## Current State

### Task Selection vs Task Creation
The curriculum system has two distinct phases:

1. **Task Selection** (already LP-based):
   - When pool is at capacity, select from existing tasks
   - Tasks sampled WITH REPLACEMENT based on individual task LP scores
   - Higher LP tasks get sampled more frequently
   - Implemented in `Curriculum._choose_task()`

2. **Task Creation** (currently uniform):
   - When pool is below capacity OR task evicted, create new task
   - TaskGenerator samples uniformly from its configuration
   - E.g., `BucketedTaskGenerator` samples buckets uniformly
   - E.g., `TaskGeneratorSet` samples by fixed weights
   - Implemented in `Curriculum._create_task()` → `TaskGenerator.get_task()`

### Current Label Tracking
Labels are already tracked in multiple places:

1. **Per-Label LP Scores** (`curriculum_env.py:259`):
   ```python
   self._per_label_lp_scores[label] = 0.99 * self._per_label_lp_scores[label] + 0.01 * lp_score
   ```
   - EMA-smoothed LP score per label (α = 0.01)
   - Updated after each task completion
   - Already logged to WandB via curriculum stats

2. **Pool Composition** (`learning_progress_algorithm.py:311`):
   ```python
   pool_composition[label] = pool_composition.get(label, 0) + 1
   ```
   - Count of each label in active task pool
   - Logged per epoch

3. **Sampling Counts** (`learning_progress_algorithm.py:240`):
   ```python
   self._label_completion_counts[label] = self._label_completion_counts.get(label, 0) + 1
   ```
   - Number of times each label was completed
   - Tracks actual training distribution

## Proposed Design

### Goal
When creating new tasks, sample from labels proportionally to their LP scores (with configurable minimum probabilities), rather than uniformly.

### Architecture Changes

#### 1. Label-Based LP Scoring System

**New Component**: `LabelLPTracker`
- Tracks per-label LP scores with EMA smoothing
- Maintains minimum sampling probability per label
- Supports training stage-based probability schedules
- Provides label weights for task generation

**Location**: `metta/cogworks/curriculum/label_lp_tracker.py`

**Interface**:
```python
class LabelLPTracker:
    """Tracks learning progress scores at the label level."""

    def __init__(
        self,
        ema_alpha: float = 0.01,
        min_sampling_prob: float = 0.05,
        training_stage_schedule: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            ema_alpha: EMA smoothing factor for label LP scores
            min_sampling_prob: Minimum sampling probability for any label
            training_stage_schedule: Optional mapping of training_stage -> min_prob
                Example: {"early": 0.15, "mid": 0.10, "late": 0.05}
        """
        pass

    def update_label_lp(self, label: str, lp_score: float) -> None:
        """Update LP score for a label with EMA smoothing."""
        pass

    def get_label_weights(
        self,
        labels: List[str],
        training_stage: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Get sampling weights for labels based on LP scores.

        Applies:
        1. LP score normalization
        2. Minimum probability floor (stage-dependent)
        3. Re-normalization to sum to 1.0

        Returns:
            Dictionary mapping label -> sampling probability
        """
        pass

    def set_training_stage(self, stage: str) -> None:
        """Update training stage for dynamic min probability."""
        pass

    def get_label_lp_score(self, label: str) -> float:
        """Get current LP score for a label."""
        pass
```

#### 2. Task Generator Modifications

**Modified**: `TaskGenerator` base class
- Add optional `sample_weights` parameter to `get_task()`
- Generators can optionally use weights when sampling
- Backward compatible (weights=None → uniform sampling)

**Modified**: `TaskGeneratorSet._generate_task()`
```python
def _generate_task(self, task_id: int, rng: random.Random, sample_weights: Optional[Dict[str, float]] = None) -> MettaGridConfig:
    """
    Args:
        sample_weights: Optional label -> weight mapping for non-uniform sampling
    """
    if sample_weights is not None:
        # Extract labels from child generators
        child_labels = [gen._config.label for gen in self._sub_task_generators]
        # Map labels to generator weights
        weights = [sample_weights.get(label, 1.0) for label in child_labels]
    else:
        weights = self._weights  # Use fixed weights (current behavior)

    chosen_generator = rng.choices(self._sub_task_generators, weights=weights)[0]
    return chosen_generator.get_task(task_id)
```

**Modified**: `BucketedTaskGenerator._generate_task()`
```python
def _generate_task(self, task_id: int, rng: random.Random, sample_weights: Optional[Dict[str, float]] = None) -> MettaGridConfig:
    """
    For bucketed generators, sample_weights could influence:
    1. Which bucket dimension to vary (if buckets are labeled)
    2. Which child generator to use (if wrapping TaskGeneratorSet)

    Default: Pass weights to child generator
    """
    pass
```

#### 3. Curriculum Integration

**Modified**: `Curriculum._create_task()`
```python
def _create_task(self) -> CurriculumTask:
    """Create a new task with a unique ID from Python's unlimited integer space."""
    task_id = self._rng.randint(0, 2**63 - 1)
    while task_id in self._task_ids:
        task_id = self._rng.randint(0, 2**63 - 1)
    self._task_ids.add(task_id)

    # NEW: Get label-based sampling weights if available
    sample_weights = None
    if self._algorithm is not None and hasattr(self._algorithm, 'get_label_sampling_weights'):
        sample_weights = self._algorithm.get_label_sampling_weights()

    # Pass weights to task generator (backward compatible)
    env_cfg = self._task_generator.get_task(task_id, sample_weights=sample_weights)

    # ... rest of existing code ...
```

**Modified**: `LearningProgressAlgorithm`
- Add `LabelLPTracker` instance
- Implement `get_label_sampling_weights()` method
- Update tracker when task performance updates occur

```python
class LearningProgressAlgorithm(CurriculumAlgorithm):
    def __init__(self, num_tasks: int, hypers: LearningProgressConfig):
        # ... existing code ...

        # NEW: Label-based LP tracking for task generation
        if hypers.use_label_lp_sampling:
            self.label_lp_tracker = LabelLPTracker(
                ema_alpha=hypers.label_lp_ema_alpha,
                min_sampling_prob=hypers.label_min_sampling_prob,
                training_stage_schedule=hypers.label_sampling_schedule,
            )
        else:
            self.label_lp_tracker = None

    def update_task_performance(self, task_id: int, score: float) -> None:
        """Update task performance using the scorer strategy."""
        # ... existing code ...

        # NEW: Update label LP tracker
        if self.label_lp_tracker is not None and task_id in self._task_labels:
            label = self._task_labels[task_id]
            task_lp_score = self.scorer.score_task(task_id, self.task_tracker)
            self.label_lp_tracker.update_label_lp(label, task_lp_score)

    def get_label_sampling_weights(self) -> Dict[str, float]:
        """Get label-based sampling weights for task generation."""
        if self.label_lp_tracker is None:
            return {}

        # Get all known labels
        labels = list(set(self._task_labels.values()))

        # Get current training stage (if tracking enabled)
        training_stage = self._get_current_training_stage()

        return self.label_lp_tracker.get_label_weights(labels, training_stage)
```

#### 4. Configuration Extensions

**Modified**: `LearningProgressConfig`
```python
class LearningProgressConfig(CurriculumAlgorithmConfig):
    """Configuration for learning progress with bidirectional scoring as default."""

    # ... existing fields ...

    # NEW: Label-based LP sampling for task generation
    use_label_lp_sampling: bool = False  # Enable label-based task generation
    label_lp_ema_alpha: float = 0.01  # EMA smoothing for label LP scores
    label_min_sampling_prob: float = 0.05  # Minimum probability per label

    # Training stage schedule (optional)
    label_sampling_schedule: Optional[Dict[str, float]] = None
    # Example: {"early": 0.15, "mid": 0.10, "late": 0.05}

    # Training stage thresholds (optional, timestep-based)
    training_stage_thresholds: Optional[Dict[str, int]] = None
    # Example: {"early": 0, "mid": 1_000_000, "late": 5_000_000}
```

### Training Stage Detection

**Strategy 1: Timestep-Based**
```python
def _get_current_training_stage(self) -> Optional[str]:
    """Determine training stage based on timesteps."""
    if self.hypers.training_stage_thresholds is None:
        return None

    current_timestep = self._get_global_timestep()  # From trainer/env

    # Find highest threshold we've passed
    stages = sorted(
        self.hypers.training_stage_thresholds.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    for stage, threshold in stages:
        if current_timestep >= threshold:
            return stage

    return "early"  # Default
```

**Strategy 2: Performance-Based**
```python
def _get_current_training_stage(self) -> Optional[str]:
    """Determine training stage based on performance metrics."""
    global_stats = self.task_tracker.get_global_stats()
    mean_score = global_stats.get("mean_score", 0.0)

    # Example thresholds
    if mean_score < 0.3:
        return "early"
    elif mean_score < 0.6:
        return "mid"
    else:
        return "late"
```

**Strategy 3: Manual (Callback-Based)**
```python
# External control via trainer
algorithm.label_lp_tracker.set_training_stage("mid")
```

## Implementation Details

### Weight Calculation Algorithm

The core algorithm in `LabelLPTracker.get_label_weights()`:

```python
def get_label_weights(
    self,
    labels: List[str],
    training_stage: Optional[str] = None,
) -> Dict[str, float]:
    """Calculate sampling weights from LP scores with minimum probability floor."""

    # Step 1: Get raw LP scores for labels
    lp_scores = {label: self._label_lp_scores.get(label, self.exploration_bonus)
                 for label in labels}

    # Step 2: Normalize LP scores to probabilities
    total_lp = sum(lp_scores.values())
    if total_lp <= 0:
        # Fallback to uniform
        return {label: 1.0 / len(labels) for label in labels}

    raw_probs = {label: lp / total_lp for label, lp in lp_scores.items()}

    # Step 3: Apply minimum probability floor (stage-dependent)
    min_prob = self._get_min_prob_for_stage(training_stage)
    adjusted_probs = {}

    for label in labels:
        adjusted_probs[label] = max(raw_probs[label], min_prob)

    # Step 4: Re-normalize to sum to 1.0
    total = sum(adjusted_probs.values())
    final_weights = {label: prob / total for label, prob in adjusted_probs.items()}

    return final_weights

def _get_min_prob_for_stage(self, stage: Optional[str]) -> float:
    """Get minimum probability for current training stage."""
    if stage is None or self.training_stage_schedule is None:
        return self.min_sampling_prob

    return self.training_stage_schedule.get(stage, self.min_sampling_prob)
```

### Example Scenarios

#### Scenario 1: Three Task Types, Early Training

Labels: `["easy", "medium", "hard"]`
LP Scores: `[0.8, 0.6, 0.4]` (easy has highest LP)
Min Prob: `0.15` (early stage)
Training Stage: `"early"`

**Calculation**:
1. Normalize LP: `[0.44, 0.33, 0.22]`
2. Apply floor: `[0.44, 0.33, 0.22]` (all above floor)
3. Re-normalize: `[0.44, 0.33, 0.22]` (already sums to 1.0)

**Result**: Easy tasks sampled 44% of the time, hard tasks 22%

#### Scenario 2: Same Setup, Late Training

LP Scores: `[0.2, 0.6, 0.9]` (hard has highest LP now)
Min Prob: `0.05` (late stage, more exploitation)
Training Stage: `"late"`

**Calculation**:
1. Normalize LP: `[0.12, 0.35, 0.53]`
2. Apply floor: `[0.12, 0.35, 0.53]` (all above floor)
3. Re-normalize: `[0.12, 0.35, 0.53]`

**Result**: Hard tasks sampled 53% of the time, easy tasks only 12%

#### Scenario 3: One Dominant Label

Labels: `["task_a", "task_b", "task_c"]`
LP Scores: `[0.95, 0.03, 0.02]` (task_a dominates)
Min Prob: `0.10`
Training Stage: `"mid"`

**Calculation**:
1. Normalize LP: `[0.95, 0.03, 0.02]`
2. Apply floor: `[0.95, 0.10, 0.10]` (b and c below floor)
3. Re-normalize: `[0.95/1.15, 0.10/1.15, 0.10/1.15]` = `[0.826, 0.087, 0.087]`

**Result**: Task A gets 82.6%, but B and C maintain ~9% each (reduced floor after normalization)

### Edge Cases & Handling

1. **No Labels Available**
   - Fallback to uniform sampling
   - No weights passed to generator

2. **Label Not Seen Before**
   - Use exploration bonus as initial LP score
   - Ensures new labels get sampled

3. **All LP Scores Zero**
   - Fallback to uniform sampling
   - Prevents division by zero

4. **Min Prob Too High** (e.g., 10 labels, min_prob=0.15)
   - After applying floor, sum > 1.0
   - Re-normalization handles this gracefully
   - Effective min_prob becomes 0.15 / 1.5 = 0.10

5. **Single Label**
   - Always returns 1.0 for that label
   - Min prob doesn't matter

6. **Negative LP Scores**
   - Clip to 0.0 before normalization
   - Or use absolute value (implementation choice)

## Migration Path

### Phase 1: Add Infrastructure (Non-Breaking)
- Implement `LabelLPTracker` class
- Add config fields (all default to False/None)
- Modify `TaskGenerator.get_task()` to accept optional `sample_weights`
- All existing code continues to work (weights=None)

### Phase 2: Integrate with Learning Progress
- Add `label_lp_tracker` to `LearningProgressAlgorithm`
- Implement `get_label_sampling_weights()`
- Update `Curriculum._create_task()` to pass weights
- Still disabled by default via config

### Phase 3: Enable for Specific Generators
- Implement weight handling in `TaskGeneratorSet`
- Implement weight handling in `BucketedTaskGenerator` (if labels per bucket)
- Add tests for weighted sampling

### Phase 4: Production Testing
- Enable via config flag in experiments
- Monitor via WandB:
  - `curriculum_stats/per_label_lp_scores`
  - `curriculum_stats/pool_composition_fraction`
  - `curriculum_stats/per_label_samples_this_epoch`
- Compare convergence vs uniform sampling

### Phase 5: Default & Tuning
- If successful, make default for LP-based curricula
- Tune default min_prob and schedules
- Document best practices

## Testing Strategy

### Unit Tests

1. **LabelLPTracker**
   - Test EMA updating
   - Test weight calculation with various LP distributions
   - Test min prob enforcement
   - Test training stage transitions
   - Test edge cases (single label, all zeros, etc.)

2. **TaskGenerator Integration**
   - Test `TaskGeneratorSet` with weights
   - Test backward compatibility (weights=None)
   - Test determinism with same seed

3. **Curriculum Integration**
   - Mock generator that records sampling calls
   - Verify weights are passed correctly
   - Verify fallback to uniform when no weights

### Integration Tests

1. **End-to-End Sampling Distribution**
   - Create curriculum with 3 labeled generators
   - Set known LP scores per label
   - Sample 10,000 tasks
   - Verify distribution matches expected probabilities (within confidence interval)

2. **Training Stage Transitions**
   - Start with early stage (high min_prob)
   - Transition to late stage (low min_prob)
   - Verify sampling distribution shifts as expected

3. **Multi-Process Consistency**
   - Multiple workers with shared memory
   - All update same label LP tracker
   - Verify convergence to consistent sampling distribution

### Performance Tests

1. **Overhead Measurement**
   - Compare task generation speed with/without label LP
   - Ensure < 5% overhead
   - Profile weight calculation bottlenecks

2. **Memory Usage**
   - Track memory growth with many labels
   - Ensure O(num_labels) not O(num_tasks)

## Monitoring & Observability

### New WandB Metrics

1. **Label LP Scores** (per label)
   - `curriculum_stats/per_label_lp_scores/{label}`: Current EMA LP score
   - Already logged by `CurriculumEnv`

2. **Label Sampling Weights** (per label)
   - `curriculum_stats/label_sampling_weights/{label}`: Current generation weight
   - NEW: Add to `CurriculumEnv._add_curriculum_stats_to_info()`

3. **Label Creation Counts** (per label)
   - `curriculum_stats/label_creation_counts/{label}`: Tasks created per label
   - NEW: Track in `Curriculum._create_task()`

4. **Sampling Distribution Metrics**
   - `curriculum_stats/label_sampling_entropy`: Entropy of label distribution
   - `curriculum_stats/label_sampling_max_ratio`: Max/min label ratio
   - Helps identify if distribution is too peaked or flat

5. **Training Stage**
   - `curriculum_stats/training_stage`: Current stage (as enum/int)
   - Track transitions over time

### Comparison Dashboards

Create WandB dashboard comparing:
- **Uniform vs LP-based sampling**
- Label distribution over time
- Convergence speed (mean reward)
- Sample efficiency (steps to threshold)
- Final performance

## Hyperparameter Tuning

### Key Parameters to Tune

1. **`label_lp_ema_alpha`** (default: 0.01)
   - Lower = slower adaptation to LP changes (more stable)
   - Higher = faster adaptation (more responsive)
   - Recommend: 0.005 - 0.05

2. **`label_min_sampling_prob`** (default: 0.05)
   - Lower = more exploitation of high-LP labels
   - Higher = more exploration of low-LP labels
   - Recommend: 0.01 - 0.15

3. **Training Stage Schedule**
   - Early: Higher min_prob (0.15) for exploration
   - Mid: Medium min_prob (0.10) for balance
   - Late: Lower min_prob (0.05) for exploitation

4. **Stage Transition Thresholds**
   - Timestep-based: Every 1-2M timesteps
   - Performance-based: Based on mean reward percentiles

### Tuning Strategies

1. **Grid Search** (small scale)
   - Test combinations on fast environment
   - Measure sample efficiency and final performance
   - Identify promising regions

2. **Progressive Exploration**
   - Start with high min_prob, gradually reduce
   - Track when performance plateaus
   - Adjust schedule accordingly

3. **Task-Specific Tuning**
   - Different task distributions need different settings
   - Many similar tasks → lower min_prob okay
   - Few diverse tasks → higher min_prob needed

## Alternative Designs Considered

### Alternative 1: Task Generator Owns LP Tracking

**Approach**: Each generator tracks LP scores internally
- Pros: Encapsulation, no curriculum changes
- Cons: Duplicated logic, harder to aggregate across generators

### Alternative 2: Global LP-Based Task Pool

**Approach**: One global pool of tasks, sampled by label LP
- Pros: Simpler conceptually
- Cons: Doesn't scale to large task spaces, loses task-level granularity

### Alternative 3: Hierarchical Sampling (Label → Task)

**Approach**: First sample label by LP, then sample task uniformly within label
- Pros: Clear two-stage process
- Cons: Loses within-label LP information, more complex

**Selected Design** balances:
- Clean separation (label LP for creation, task LP for selection)
- Minimal API changes (optional weights parameter)
- Backward compatibility (disabled by default)
- Performance (computed per label, not per task)

## Open Questions

1. **Should BucketedTaskGenerator support per-bucket weighting?**
   - Currently, buckets don't have labels
   - Could add label per bucket value/range
   - Or pass through to child generator

2. **How to handle hierarchical labels?**
   - E.g., "easy/combat" vs "easy/navigation"
   - Could aggregate LP at different levels
   - Or require unique labels

3. **Should min_prob be per-label or global?**
   - Current design: global (same for all labels)
   - Alternative: Dict[label, min_prob] for fine control
   - Trade-off: simplicity vs flexibility

4. **How to handle label changes during training?**
   - If generators add new labels dynamically
   - Need to initialize LP scores appropriately
   - Use exploration bonus as default

5. **Should we support curriculum-level overrides?**
   - Manual weight injection for debugging
   - Useful for ablation studies
   - API: `curriculum.set_label_weights({"easy": 0.5, ...})`

## Success Criteria

This feature will be considered successful if:

1. **Performance**: Sample efficiency improves by >10% vs uniform sampling
2. **Flexibility**: Min prob schedules enable different training strategies
3. **Stability**: No regression in final performance
4. **Usability**: Config-based enable/disable, clear defaults
5. **Observability**: WandB dashboards show clear label distribution dynamics
6. **Compatibility**: Existing curricula work unchanged (backward compatible)

## Future Extensions

1. **Multi-Armed Bandit Approaches**
   - UCB-style exploration bonuses per label
   - Thompson sampling for label selection
   - Contextual bandits with curriculum state

2. **Adaptive Min Prob**
   - Automatically adjust based on label diversity
   - Higher min_prob when labels have high variance
   - Lower min_prob when labels are homogeneous

3. **Cross-Label Dependencies**
   - Model prerequisite relationships (easy → medium → hard)
   - Weight based on "readiness" not just LP
   - Requires curriculum graph structure

4. **Population-Based Training for Curriculum**
   - Treat min_prob schedules as hyperparameters
   - Evolve schedules across training runs
   - Discover optimal trajectories

## Implementation Timeline

- **Week 1-2**: Implement `LabelLPTracker` + tests
- **Week 3-4**: Integrate with `LearningProgressAlgorithm` + tests
- **Week 5-6**: Modify task generators + tests
- **Week 7-8**: End-to-end integration + tests
- **Week 9-10**: Production testing on real curriculum
- **Week 11-12**: Hyperparameter tuning + documentation
- **Week 13+**: Rollout, monitoring, iteration

## References

- Current LP implementation: `learning_progress_algorithm.py`
- Task generation: `task_generator.py`
- Label tracking: `curriculum_env.py:259`
- Pool composition stats: `learning_progress_algorithm.py:311`

## Document Metadata

- **Created**: 2025-10-22
- **Author**: AI Assistant (via user request)
- **Status**: Design Proposal
- **Version**: 1.0

