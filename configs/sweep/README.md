# Sweep Configuration Structure

This directory contains sweep configurations for hyperparameter optimization using the Protein optimizer.

## New Structure (Recommended)

As of the latest update, sweep configurations follow a cleaner structure that separates sweep metadata from training configuration overrides:

### Sweep Configuration Files (in `configs/sweep/`)

These files contain:
1. **Sweep metadata** - Information for the Protein optimizer
2. **Parameter search space** - Hyperparameters to sweep over

Example structure:
```yaml
# Sweep metadata
metric: "episode/reward.mean"  # Metric to optimize
goal: maximize                 # Direction: maximize or minimize
num_random_samples: 10         # Random samples before Bayesian optimization

# Parameter search space
parameters:
  trainer.learning_rate: ${ss:log, 1e-5, 1e-3}
  trainer.batch_size: ${ss:pow2, 2048, 8192}
  agent.components._core_.hidden_size: ${ss:pow2, 128, 512}
```

### Sweep Job Configuration (in `configs/sweep_job.yaml`)

The `sweep_job` section contains configuration overrides that apply to ALL runs in the sweep:

```yaml
sweep_job:
  # Fixed training configuration for all runs
  trainer:
    evaluate_interval: 300
    checkpoint_interval: 1000

  # Fixed agent configuration
  agent:
    clip_range: 0.2

  # Evaluation configuration
  sim:
    num_episodes: 10
```

## Key Differences from Old Structure

**Old structure** (deprecated):
- Mixed sweep metadata with config overrides in the same section
- Required filtering out metadata keys like "metric", "num_random_samples"
- Less clear separation of concerns

**New structure**:
- Clean separation: `sweep/` files for optimizer metadata, `sweep_job` for config overrides
- No ambiguity about what's being swept vs what's fixed
- Easier to understand and maintain

## Available Sweep Configurations

- `fast.yaml` - Quick sweep with small parameter ranges for testing
- `full.yaml` - Comprehensive sweep with full parameter ranges
- `example_new_structure.yaml` - Example demonstrating the new structure

## Parameter Space Syntax

Parameters use the `${ss:...}` syntax for defining search spaces:

- `${ss:log, min, max}` - Log-uniform distribution
- `${ss:logit, min, max}` - Logit-uniform distribution (for probabilities)
- `${ss:linear, min, max}` - Uniform distribution
- `${ss:int, min, max}` - Integer uniform distribution
- `${ss:pow2, min, max}` - Powers of 2 (for batch sizes, hidden dimensions)
