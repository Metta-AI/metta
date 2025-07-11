# Bug Report: Interdependent Parameter Constraints in Sweep System

**Priority:** Medium-High
**Component:** Sweep System / Hyperparameter Optimization
**Status:** Active Issue

## Problem Description

The sweep system allows independent sweeping of interdependent hyperparameters (batch_size, minibatch_size, bptt_horizon, total_agents) without validating their mathematical constraints. This leads to training crashes that waste computational resources and create misleading failure signals for the Protein optimizer.

## Current Behavior

### Interdependent Constraints
The system has several hard mathematical constraints between parameters:

1. **Batch Size Divisibility**: `batch_size % minibatch_size == 0`
2. **BPTT Horizon Divisibility**: `minibatch_size % bptt_horizon == 0`
3. **Capacity Constraints**: `batch_size <= total_agents * buffer_size`

### Failure Mode
When sweep parameters violate these constraints:
1. Training starts with invalid parameter combinations
2. Trainer crashes during initialization with `ValueError`
3. Sweep run marked as "failed"
4. Protein learns from these artificial failures
5. Computational resources wasted on doomed configurations

## Impact Assessment

### Resource Waste
- **Computational Cost**: Failed runs consume cluster resources for initialization
- **Time Cost**: Each failed run takes time to start, crash, and clean up
- **Opportunity Cost**: Resources could be used for viable configurations

### Optimizer Confusion
- **Misleading Signals**: Protein receives failure signals for mathematical impossibilities, not actual hyperparameter performance
- **Search Space Pollution**: Invalid regions contaminate the optimization landscape
- **Convergence Issues**: Optimizer may avoid valid nearby regions due to constraint violations

## Technical Details

### Constraint Validation Locations
```python
# metta/rl/trainer_config.py:224-225
if self.batch_size % self.minibatch_size != 0:
    raise ValueError("batch_size must be divisible by minibatch_size")

# metta/rl/experience.py:119
if self.minibatch_size % bptt_horizon != 0:
    raise ValueError(f"minibatch_size {self.minibatch_size} must be divisible by bptt_horizon {bptt_horizon}")
```

### Current Sweep Configuration Example
```yaml
# configs/sweep/full.yaml - Can create invalid combinations
parameters:
  trainer.batch_size:
    values: [128, 256, 512, 1024]
  trainer.minibatch_size:
    values: [16, 32, 64, 128]
  trainer.bptt_horizon:
    values: [8, 16, 32]
```

**Problem**: This creates 4×4×3 = 48 combinations, but many are invalid:
- `batch_size=128, minibatch_size=64, bptt_horizon=32` → Invalid (64 % 32 ≠ 0 is false, but 128 % 64 = 0 is true)
- `batch_size=256, minibatch_size=128, bptt_horizon=32` → Invalid (128 % 32 ≠ 0 is false)

## Reproduction Steps

1. Create a sweep config with conflicting parameters:
```yaml
parameters:
  trainer.batch_size:
    values: [100]
  trainer.minibatch_size:
    values: [7]
  trainer.bptt_horizon:
    values: [13]
```

2. Run sweep: `./devops/skypilot/launch.py sweep run=test_constraints`
3. Observe training crash with `ValueError: batch_size must be divisible by minibatch_size`
4. Check WandB - run marked as "failed"

## Proposed Solutions

### Option 1: Constraint Resolution in Sweep Init (Recommended)
- **Location**: Modify `tools/sweep_init.py`
- **Approach**: Validate and fix parameter combinations before sending to Protein
- **Benefits**: Minimal infrastructure changes, guarantees valid configurations
- **Implementation**: Add constraint resolver that adjusts dependent parameters

### Option 2: Protein Constraint Integration
- **Location**: Modify `metta/sweep/protein.py`
- **Approach**: Add constraint validation to suggestion generation
- **Benefits**: Optimizer-aware constraint handling
- **Drawbacks**: More complex, requires Protein architecture changes

### Option 3: Smart Parameter Coupling
- **Location**: Sweep configuration system
- **Approach**: Define parameter relationships in sweep configs
- **Benefits**: Declarative constraint specification
- **Drawbacks**: Requires new configuration syntax

## Recommended Implementation

Implement **Option 1** with constraint resolution in sweep initialization:

```python
def resolve_parameter_constraints(params):
    """Resolve interdependent parameter constraints"""
    # Ensure batch_size divisible by minibatch_size
    if params.get('trainer.batch_size') and params.get('trainer.minibatch_size'):
        batch_size = params['trainer.batch_size']
        minibatch_size = params['trainer.minibatch_size']
        if batch_size % minibatch_size != 0:
            # Adjust minibatch_size to largest valid divisor
            params['trainer.minibatch_size'] = find_largest_divisor(batch_size, minibatch_size)

    # Ensure minibatch_size divisible by bptt_horizon
    if params.get('trainer.minibatch_size') and params.get('trainer.bptt_horizon'):
        minibatch_size = params['trainer.minibatch_size']
        bptt_horizon = params['trainer.bptt_horizon']
        if minibatch_size % bptt_horizon != 0:
            # Adjust bptt_horizon to largest valid divisor
            params['trainer.bptt_horizon'] = find_largest_divisor(minibatch_size, bptt_horizon)

    return params
```

## Files to Modify

1. **`tools/sweep_init.py`** - Add constraint resolution
2. **`metta/sweep/protein_metta.py`** - Call constraint resolver before parameter application
3. **`tests/sweep/test_constraints.py`** - Add constraint validation tests

## Testing Strategy

1. **Unit Tests**: Test constraint resolution logic with various parameter combinations
2. **Integration Tests**: Verify sweep runs complete without constraint violations
3. **Regression Tests**: Ensure existing valid configurations still work
4. **Performance Tests**: Measure overhead of constraint resolution

## Priority Justification

**Medium-High Priority** because:
- **Resource Efficiency**: Prevents wasted computational resources
- **Optimizer Quality**: Improves Protein's learning signal quality
- **User Experience**: Reduces confusing failed runs
- **System Reliability**: Makes sweep system more robust

## Related Issues

- WandB Sweep States Bug (perpetual "running" status)
- Evaluation System Bug (hardcoded reward metric)
- Missing metadata score bug (random policy selection)

---

**Reporter**: AI Assistant
**Date**: 2024-12-28
**Affected Versions**: Current main branch
