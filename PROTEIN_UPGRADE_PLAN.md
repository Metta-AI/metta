# Protein Advanced Upgrade Plan

## Overview

This document outlines all changes needed to upgrade the Metta sweep pipeline from the current `protein.py` to the new
`protein_advanced.py` implementation.

## Key Improvements in ProteinAdvanced

1. **Proper acquisition functions** (Expected Improvement, Upper Confidence Bound)
2. **Better GP handling** with dimension safety
3. **Multi-objective support** (ready for future use)
4. **Cleaner API** with ObservationPoint dataclass
5. **Better Pareto frontier computation** (O(n log n) for 2D)

## Required Changes

### 1. Update `metta/sweep/__init__.py`

```python
# Add new imports
from .protein_advanced import ProteinAdvanced, ObservationPoint, efficient_pareto_points
```

### 2. Update `metta/sweep/protein_metta.py`

Replace the current Protein import and usage with ProteinAdvanced:

```python
from .protein_advanced import ProteinAdvanced

class MettaProtein:
    def __init__(self, cfg: DictConfig):
        self._cfg = cfg

        # Convert OmegaConf to regular dict
        parameters_dict = OmegaConf.to_container(cfg.parameters, resolve=True)

        # Create the config structure that ProteinAdvanced expects
        protein_config = {
            "metric": cfg.metric,
            "goal": cfg.goal,
            "method": cfg.method,
            **parameters_dict,  # Add flattened parameters at top level
        }

        # Extract protein settings, with new defaults for ProteinAdvanced
        protein_settings = OmegaConf.to_container(cfg.protein, resolve=True) if 'protein' in cfg else {}

        # Map old parameter names to new ones if needed
        if 'resample_frequency' in protein_settings:
            # resample_frequency doesn't exist in ProteinAdvanced, remove it
            del protein_settings['resample_frequency']

        # Set default acquisition function if not specified
        if 'acquisition_fn' not in protein_settings:
            protein_settings['acquisition_fn'] = 'ei'  # Use Expected Improvement by default

        # Initialize ProteinAdvanced with sweep config and settings
        self._protein = ProteinAdvanced(protein_config, **protein_settings)

    @property
    def num_observations(self) -> int:
        """Get the number of observations."""
        # ProteinAdvanced uses a single observations list
        return len(self._protein.observations)
```

### 3. Update Sweep Configuration Files

#### Update `configs/sweep/full.yaml` and phase configs:

```yaml
protein:
  # Core ProteinAdvanced parameters
  acquisition_fn: ei # New: 'ei' or 'ucb'
  num_random_samples: 20 # Same as before
  max_suggestion_cost: 3600 # Same as before
  # resample_frequency: REMOVE THIS (not in ProteinAdvanced)
  global_search_scale: 1.0 # Same as before
  random_suggestions: 256 # Reduced from 1024 in advanced
  suggestions_per_pareto: 64 # Reduced from 256 in advanced
  expansion_rate: 0.15 # Same as before
  seed_with_search_center: true # Same as before

  # New optional parameters (with defaults)
  beta_ucb: 2.0 # For UCB acquisition function
  xi_ei: 0.01 # For EI acquisition function

# Rest of config remains the same
metric: heart.gained # or whatever metric you're using
goal: maximize
method: bayes
```

#### Update phase configs similarly:

- `configs/sweep/phase1_explore_cheap.yaml`
- `configs/sweep/phase2_exploit_medium.yaml`
- `configs/sweep/phase3_final_expensive.yaml`

Remove `resample_frequency` from all configs and add `acquisition_fn: ei`.

### 4. Update `metta/sweep/wandb_utils.py` (if needed)

Check if any functions directly access `protein.success_observations` or `protein.failure_observations`. These need to
be updated to use `protein.observations` instead:

```python
# Old way
success_count = len(protein.success_observations)
failure_count = len(protein.failure_observations)

# New way
success_count = len([obs for obs in protein.observations if not obs.is_failure])
failure_count = len([obs for obs in protein.observations if obs.is_failure])
```

### 5. Backward Compatibility

The `protein_advanced.py` includes a backward-compatible `Protein` class that wraps `ProteinAdvanced`. This means
existing code can continue to work, but for best results, use `ProteinAdvanced` directly.

### 6. Testing Strategy

1. **Unit tests**: Update tests in `tests/sweep/` to use ProteinAdvanced
2. **Integration test**: Run a small sweep with the new implementation
3. **Comparison test**: Run identical sweeps with old and new implementations to verify improvements

## Migration Path

### Option 1: Gradual Migration (Recommended)

1. Add `protein_advanced.py` to the codebase ✅ (already done)
2. Create `protein_metta_advanced.py` that uses ProteinAdvanced
3. Update configs to support both old and new protein implementations
4. Run side-by-side comparisons
5. Once validated, switch default to ProteinAdvanced
6. Eventually deprecate old implementation

### Option 2: Direct Replacement

1. Update `protein_metta.py` to use ProteinAdvanced directly
2. Update all configs at once
3. Run comprehensive tests
4. Deploy

## Configuration Examples

### Minimal Config (using defaults)

```yaml
protein:
  acquisition_fn: ei
  num_random_samples: 10
  max_suggestion_cost: 3600

metric: reward
goal: maximize
method: bayes
```

### Advanced Config (with all options)

```yaml
protein:
  acquisition_fn: ucb # or 'ei'
  num_random_samples: 20
  max_suggestion_cost: 7200
  global_search_scale: 1.0
  random_suggestions: 256
  suggestions_per_pareto: 64
  seed_with_search_center: true
  expansion_rate: 0.25
  beta_ucb: 3.0 # Higher for more exploration (only for UCB)
  xi_ei: 0.01 # Higher for more exploration (only for EI)

metric: heart.gained
goal: maximize
method: bayes
```

## Benefits of Upgrading

1. **Better convergence**: Proper acquisition functions (EI/UCB) vs random acquisition
2. **More stable**: Fixed dimension handling and Pareto computation bugs
3. **Future-ready**: Built-in support for constraints and multi-fidelity (not enabled yet)
4. **Cleaner code**: Better separation of concerns and cleaner API
5. **Performance**: O(n log n) Pareto frontier for 2D objectives

## Rollback Plan

If issues arise:

1. The old `protein.py` remains unchanged
2. Simply revert the import in `protein_metta.py`
3. Revert config changes (mainly removing `acquisition_fn` parameter)

## Validation Checklist

- [x] ProteinAdvanced passes all existing tests
- [ ] Sweep runs complete successfully with new implementation
- [ ] Performance metrics show improvement over baseline
- [ ] No memory leaks or performance regressions
- [ ] Config validation works properly
- [ ] WandB integration continues to work
