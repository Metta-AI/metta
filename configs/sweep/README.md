# Sweep Configurations

This directory contains hyperparameter sweep configurations for the Metta training system using the Protein optimizer (Gaussian Process-based optimization).

## Overview

The sweep system automates hyperparameter optimization through:
1. Define parameter search space in YAML configuration
2. Run sweep with `./devops/sweep.sh`
3. Protein optimizer suggests parameters
4. System trains with suggested parameters
5. Results feed back to optimizer

## Configuration Format

Sweep configurations use YAML format with parameters defined under a `sweep` section:

```yaml
# Required: Parameter search space
parameters:
  trainer.learning_rate:
    min: 0.00001
    max: 0.01
    distribution: log_normal  # Options: uniform, log_normal, int_uniform

  trainer.batch_size:
    min: 32
    max: 256
    distribution: int_uniform

# Optional: Optimization settings
metric: reward        # Default: reward
goal: maximize       # Default: maximize

# Optional: Additional settings
num_random_samples: 10  # Initial random samples before optimization
```

### Parameter Distributions

| Distribution | Use Case | Example |
|-------------|----------|---------|
| `uniform` | Linear ranges | gamma: [0.9, 0.999] |
| `log_normal` | Exponential ranges | learning_rate: [1e-5, 1e-2] |
| `int_uniform` | Integer ranges | batch_size: [32, 256] |

## Usage

### Running a Sweep

```bash
# Basic usage
./devops/sweep.sh run=my_sweep ++sweep_params=sweep/fast --rollout-count=10

# With hardware configuration
./devops/sweep.sh run=my_sweep ++sweep_params=sweep/complex +hardware=aws --rollout-count=50
```

### Validating Configuration

```bash
# Validate before running
python -m metta.rl.protein_opt.sweep_config configs/sweep/my_sweep.yaml -v
```

## Available Configurations

- **`fast_30s.yaml`** - Ultra-fast test sweep (30 seconds)
- **`fast.yaml`** - Quick parameter sweep
- **`complex.yaml`** - Comprehensive 10-parameter optimization
- **`full.yaml`** - Full parameter search space
- **`cogeval_sweep.yaml`** - Cognitive evaluation tasks

## Creating Custom Sweeps

1. Create a new YAML file in `configs/sweep/`
2. Define parameters under the `parameters` section
3. Set appropriate min/max ranges and distributions
4. Validate with the validation tool
5. Run with `./devops/sweep.sh`

Example:
```yaml
parameters:
  trainer.learning_rate:
    min: 0.0001
    max: 0.01
    distribution: log_normal

  trainer.gamma:
    min: 0.95
    max: 0.999
    distribution: uniform
```

## Best Practices

- Start with few parameters (3-4) before scaling up
- Use `log_normal` distribution for learning rates
- Ensure batch sizes are divisible by minibatch sizes
- Set appropriate `--rollout-count` based on parameter space size
- Monitor progress via WandB dashboard

## Troubleshooting

**Validation errors**: Run the validation tool to check configuration syntax

**Memory issues**: Reduce batch_size ranges or adjust minibatch_size

**Sweep not converging**: Increase `num_random_samples` for better initial exploration

## Migration from CARBS

If migrating from old CARBS configs:

| CARBS Syntax | Protein Equivalent |
|--------------|-------------------|
| `${ss:log, 1e-5, 1e-2}` | `min: 0.00001, max: 0.01, distribution: log_normal` |
| `${ss:int, 1, 10}` | `min: 1, max: 10, distribution: int_uniform` |

## Related Documentation

- [Protein Optimizer Paper](https://arxiv.org/abs/2308.00352)
- [WandB Documentation](https://docs.wandb.ai/guides/sweeps)
