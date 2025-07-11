# Feature Request: Intelligent Sweep Termination Criteria

**Type:** Enhancement
**Priority:** High
**Component:** Sweep System / Hyperparameter Optimization
**Status:** Proposed

## Problem Statement

Currently, sweeps run indefinitely until manually stopped or timeout is reached. There are no intelligent termination criteria based on optimization progress, convergence detection, or diminishing returns. This leads to significant resource waste and suboptimal sweep efficiency.

### Current Behavior
- Sweeps continue until `max_suggestion_cost` timeout per run
- No automatic termination based on optimization progress
- No early stopping for clearly poor hyperparameter regions
- Manual intervention required to stop unproductive sweeps
- No convergence detection across the parameter space

### Impact
- **Resource Waste**: Continued exploration after optimal regions are found
- **Opportunity Cost**: Resources spent on diminishing returns instead of new experiments
- **Manual Overhead**: Requires constant monitoring and manual termination
- **Suboptimal Results**: May miss optimal stopping points or continue past convergence

## Proposed Solution

Implement **intelligent sweep termination criteria** that automatically detect when further exploration is unlikely to yield significant improvements, while providing multiple termination strategies for different use cases.

### Core Termination Strategies

#### 1. **Convergence-Based Termination**
Detect when the optimization has converged to a stable region:
- **Plateau Detection**: No improvement in best score for N consecutive runs
- **Variance Threshold**: Recent runs show low variance in performance
- **Improvement Rate**: Rate of improvement falls below threshold

#### 2. **Budget-Based Termination**
Intelligent resource allocation and stopping:
- **Total Budget Limit**: Maximum total compute time/cost across all runs
- **Diminishing Returns**: Stop when cost per unit improvement exceeds threshold
- **ROI Threshold**: Return on investment falls below acceptable level

#### 3. **Statistical Termination**
Statistically-informed stopping criteria:
- **Confidence Intervals**: Best configuration confidence interval stabilizes
- **Sample Size**: Sufficient samples to make statistically valid conclusions
- **Exploration Completeness**: Adequate coverage of parameter space

#### 4. **Performance-Based Termination**
Stop based on absolute performance achievements:
- **Target Score**: Terminate when target performance is reached
- **Percentile Achievement**: Stop when Nth percentile performance is achieved
- **Baseline Comparison**: Terminate when significantly better than baseline

## Technical Implementation

### 1. **Termination Configuration**

Add termination criteria to sweep configuration:

```yaml
# configs/sweep/full.yaml
termination:
  enabled: true

  # Convergence-based criteria
  convergence:
    plateau_patience: 10          # Stop after 10 runs without improvement
    improvement_threshold: 0.01   # Minimum improvement to count as progress
    variance_threshold: 0.05      # Stop when recent variance < 5%
    lookback_window: 20           # Consider last 20 runs for convergence

  # Budget-based criteria
  budget:
    max_total_cost: 36000        # 10 hours total across all runs
    max_runs: 100                # Maximum number of runs
    roi_threshold: 0.1           # Stop when ROI < 10%

  # Performance-based criteria
  performance:
    target_score: 0.95           # Stop when target reached
    baseline_improvement: 0.2    # Stop when 20% better than baseline

  # Statistical criteria
  statistical:
    confidence_level: 0.95       # 95% confidence for best configuration
    min_samples: 20              # Minimum runs before considering termination
```

### 2. **Termination Engine**

Create a dedicated termination decision system:

```python
# metta/sweep/termination.py
class SweepTerminationEngine:
    def __init__(self, config: TerminationConfig):
        self.config = config
        self.run_history = []
        self.best_score = None
        self.plateau_count = 0

    def should_terminate(self, current_run_result: RunResult) -> Tuple[bool, str]:
        """
        Determine if sweep should terminate based on current results.

        Returns:
            (should_terminate, reason)
        """
        self.run_history.append(current_run_result)

        # Check each termination criterion
        for criterion in self.config.criteria:
            if criterion.check(self.run_history):
                return True, criterion.reason

        return False, ""

    def get_termination_summary(self) -> Dict[str, Any]:
        """Generate summary of termination decision and sweep progress."""
        return {
            "total_runs": len(self.run_history),
            "best_score": self.best_score,
            "plateau_count": self.plateau_count,
            "estimated_remaining_improvement": self._estimate_remaining_improvement(),
            "resource_efficiency": self._calculate_resource_efficiency(),
        }
```

### 3. **Integration with Sweep System**

Modify sweep orchestration to check termination criteria:

```python
# metta/sweep/protein_metta.py
class MettaProtein:
    def __init__(self, config, wandb_run):
        # ... existing init ...
        self.termination_engine = SweepTerminationEngine(config.termination)

    def suggest(self) -> Tuple[Dict[str, Any], bool]:
        """
        Generate next hyperparameter suggestion.

        Returns:
            (suggestion, should_continue)
        """
        # Check termination criteria before suggesting
        if self.termination_engine.should_terminate(self.last_result):
            return {}, False

        # ... existing suggestion logic ...
        return suggestion, True
```

### 4. **Sweep Controller Updates**

Update sweep rollout to handle termination:

```bash
# devops/sweep_rollout.sh
while true; do
  # Check if sweep should terminate
  if ./tools/sweep_termination_check.py dist_cfg_path=$DIST_CFG_PATH; then
    echo "[TERMINATE] Sweep termination criteria met"
    break
  fi

  # Continue with normal rollout
  # ... existing rollout logic ...
done
```

## Termination Criteria Details

### 1. **Plateau Detection**
```python
class PlateauTermination:
    def check(self, run_history: List[RunResult]) -> bool:
        if len(run_history) < self.min_runs:
            return False

        recent_best = max(run_history[-self.patience:], key=lambda x: x.score)
        historical_best = max(run_history[:-self.patience], key=lambda x: x.score)

        improvement = recent_best.score - historical_best.score
        return improvement < self.threshold
```

### 2. **Variance-Based Termination**
```python
class VarianceTermination:
    def check(self, run_history: List[RunResult]) -> bool:
        if len(run_history) < self.window_size:
            return False

        recent_scores = [r.score for r in run_history[-self.window_size:]]
        variance = np.var(recent_scores)

        return variance < self.threshold
```

### 3. **ROI-Based Termination**
```python
class ROITermination:
    def check(self, run_history: List[RunResult]) -> bool:
        if len(run_history) < 2:
            return False

        recent_improvement = self._calculate_recent_improvement(run_history)
        recent_cost = self._calculate_recent_cost(run_history)

        roi = recent_improvement / recent_cost if recent_cost > 0 else float('inf')
        return roi < self.threshold
```

## Benefits

### Resource Efficiency
- **Automatic Stopping**: Eliminates need for manual sweep monitoring
- **Cost Optimization**: Prevents resource waste on diminishing returns
- **Intelligent Allocation**: Focuses compute on productive exploration

### Scientific Rigor
- **Statistical Validity**: Ensures sufficient sampling for reliable conclusions
- **Convergence Guarantees**: Provides confidence in optimization results
- **Reproducible Stopping**: Consistent termination criteria across experiments

### User Experience
- **Hands-Off Operation**: Reduces manual intervention requirements
- **Transparent Decisions**: Clear reasoning for termination decisions
- **Configurable Strategies**: Flexible termination criteria for different use cases

## Implementation Strategy

### Phase 1: Core Termination Logic
1. Implement basic termination criteria (plateau, budget, target score)
2. Add termination engine to sweep system
3. Create configuration schema for termination settings

### Phase 2: Advanced Criteria
1. Add statistical termination criteria
2. Implement variance-based and ROI-based termination
3. Add termination prediction and early warnings

### Phase 3: Intelligence & Optimization
1. Add machine learning-based termination prediction
2. Implement adaptive termination thresholds
3. Add termination recommendation system

## Configuration Examples

### Conservative Termination (Research)
```yaml
termination:
  convergence:
    plateau_patience: 20
    improvement_threshold: 0.005
  budget:
    max_runs: 200
  statistical:
    confidence_level: 0.99
```

### Aggressive Termination (Development)
```yaml
termination:
  convergence:
    plateau_patience: 5
    improvement_threshold: 0.02
  budget:
    max_runs: 50
  performance:
    target_score: 0.8
```

### Budget-Conscious Termination (Production)
```yaml
termination:
  budget:
    max_total_cost: 7200  # 2 hours
    roi_threshold: 0.2
  performance:
    baseline_improvement: 0.1
```

## Testing Strategy

### Unit Tests
- Test individual termination criteria logic
- Validate termination decision accuracy
- Test edge cases and boundary conditions

### Integration Tests
- Test termination integration with sweep system
- Verify WandB logging of termination events
- Test termination with different sweep configurations

### Performance Tests
- Measure termination detection overhead
- Test termination with large run histories
- Validate termination prediction accuracy

## Success Metrics

### Efficiency Metrics
- **Resource Savings**: Reduction in total sweep compute time
- **Termination Accuracy**: Percentage of sweeps terminated at optimal points
- **Time to Convergence**: Reduction in time to find optimal configurations

### Quality Metrics
- **Optimization Quality**: Final sweep results compared to manual termination
- **False Positive Rate**: Sweeps terminated too early
- **False Negative Rate**: Sweeps that should have terminated but didn't

## Risk Assessment

### Medium Risk Areas
- **Premature Termination**: Risk of stopping before finding optimal solutions
- **Configuration Complexity**: Many termination parameters to tune
- **Computational Overhead**: Termination checking adds processing cost

### Mitigation Strategies
- **Conservative Defaults**: Start with conservative termination thresholds
- **Termination Warnings**: Warn before termination with option to continue
- **Comprehensive Testing**: Extensive validation on historical sweep data
- **Gradual Rollout**: Optional feature with manual override capability

## Alternative Approaches Considered

### Simple Timeout-Based
- **Pros**: Simple implementation, predictable behavior
- **Cons**: Ignores optimization progress, may waste resources

### Manual Termination Only
- **Pros**: Full user control, no false positives
- **Cons**: Requires constant monitoring, inconsistent decisions

### Fixed Run Count
- **Pros**: Predictable resource usage
- **Cons**: May stop too early or too late, ignores convergence

## Files to Modify

1. **`metta/sweep/termination.py`** - Core termination engine (new file)
2. **`metta/sweep/protein_metta.py`** - Integration with Protein optimizer
3. **`devops/sweep_rollout.sh`** - Termination checking in rollout loop
4. **`tools/sweep_termination_check.py`** - Termination checking utility (new file)
5. **`configs/sweep/full.yaml`** - Add termination configuration
6. **`tests/sweep/test_termination.py`** - Comprehensive termination tests (new file)

## Acceptance Criteria

- [ ] Sweeps automatically terminate when plateau is detected
- [ ] Budget-based termination prevents resource overruns
- [ ] Statistical termination ensures sufficient sampling
- [ ] Performance-based termination stops when targets are met
- [ ] Termination decisions are logged and explainable
- [ ] Configuration allows flexible termination strategies
- [ ] Integration works with existing sweep infrastructure
- [ ] Comprehensive test coverage for all termination scenarios
- [ ] Performance metrics show measurable resource savings
- [ ] False positive rate for premature termination < 5%

---

**Requested by**: User
**Date**: 2024-12-28
**Estimated Effort**: 4-5 developer days
**Dependencies**: None (extends existing sweep system)
