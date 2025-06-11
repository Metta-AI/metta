# Sweep Configurations

This directory contains hyperparameter sweep configurations for the Metta training system using the **Protein optimizer** (Gaussian Process-based optimization).

## 🔄 **CARBS → Protein Migration Complete**

All sweep configs now use the **Protein format** with automatic rollout limiting. The old CARBS `${ss:...}` syntax has been replaced with proper parameter distributions.

## Available Configurations

### Quick Testing & Development
- **`protein_lightning.yaml`** ⚡ - Ultra-fast sweep (3 rollouts, ~9 minutes)
  - 1 parameter: learning_rate
  - 5 timesteps per run (~3 mins each)
  - Usage: `./test_lightning_sweep.sh`

- **`protein_working.yaml`** 🎯 - Standard test sweep (5 rollouts, ~15-20 minutes)
  - 3 parameters: learning_rate, gamma, batch_size
  - 10 timesteps per run (~3-4 mins each)
  - Usage: `./devops/sweep.sh run=test ++sweep_params=sweep/protein_working +hardware=macbook`

- **`protein_fast.yaml`** 🏃 - Fast comprehensive sweep (10 rollouts, ~1-2 hours)
  - 8 parameters: learning_rate, gamma, gae_lambda, vf_coef, ent_coef, batch_size, bptt_horizon, altar.cooldown
  - 5K timesteps per run (~10-15 mins each)

### Converted CARBS Configs (New Protein Format)
- **`fast_protein.yaml`** 🔄 - Converted from fast.yaml
- **`full_protein.yaml`** 🔄 - Converted from full.yaml
- **`my_sweep.yaml`** 🔄 - Converted template for customization

### Legacy CARBS Configs (Deprecated)
- **`fast.yaml`** ⚠️ - Legacy CARBS format (use `fast_protein.yaml` instead)
- **`full.yaml`** ⚠️ - Legacy CARBS format (use `full_protein.yaml` instead)

### Production Sweeps
- **`protein_simple.yaml`** 🧪 - Simple 4-parameter optimization
  - Core hyperparameters: learning_rate, batch_size, gamma, clip_param
  - Good starting point for most environments

- **`protein_complex.yaml`** 🧬 - Complex 10+ parameter optimization
  - Comprehensive parameter space (~10^23 combinations)
  - For thorough optimization of well-understood environments

### Environment-Specific
- **`pong.yaml`** 🏓 - Optimized for Pong environment
- **`cogeval_sweep.yaml`** 🧠 - Cognitive evaluation tasks
- **`full.yaml`** 📊 - Comprehensive parameter space
- **`my_sweep.yaml`** 👤 - User customization template

### Empty Templates
- **`empty.yaml`** 📝 - Empty template for new sweeps

## Usage Patterns

### Quick Testing (30 minutes)
```bash
# Test sweep infrastructure
./test_quick_sweep.sh

# Or manually:
./devops/sweep.sh run=quick_test ++sweep_params=sweep/quick_sweep +hardware=macbook trainer.num_workers=1
```

### Development (1-2 hours)
```bash
./devops/sweep.sh run=dev_test ++sweep_params=sweep/demo_sweep +hardware=macbook
```

### Production (4-8 hours)
```bash
./devops/sweep.sh run=production ++sweep_params=sweep/protein_simple +hardware=aws
```

### Full Optimization (8+ hours)
```bash
./devops/sweep.sh run=full_opt ++sweep_params=sweep/protein_complex +hardware=aws
```

## Configuration Format

### New Protein Format (Recommended)

```yaml
# @package _global_
# Rollout limiting (replaces infinite CARBS sweeps)
rollout_count: 10  # Stop after 10 experiments

sweep:
  parameters:
    # Parameter definitions with distributions
    trainer.learning_rate:
      min: 0.00001
      max: 0.01
      mean: 0.001
      scale: 1
      distribution: log_normal

    trainer.batch_size:
      min: 32
      max: 256
      mean: 64
      scale: 1
      distribution: int_uniform

  # Protein optimizer settings
  metric: reward
  goal: maximize

# Training overrides
trainer:
  total_timesteps: 50000
  evaluate_interval: 5000
```

### Legacy CARBS Format (Deprecated)

```yaml
# Old format - DO NOT USE for new sweeps
trainer:
  optimizer:
    learning_rate: ${ss:log, 1e-5, 1e-2}  # ❌ Deprecated
  batch_size: ${ss:pow2, 32, 256}         # ❌ Deprecated
```

### Migration Guide

**CARBS → Protein conversion:**
- `${ss:log, min, max}` → `distribution: log_normal, min: X, max: Y`
- `${ss:logit, min, max}` → `distribution: uniform, min: X, max: Y`
- `${ss:pow2, min, max}` → `distribution: int_uniform, min: X, max: Y`
- `${ss:int, min, max}` → `distribution: int_uniform, min: X, max: Y`

## Hardware Configurations

Always specify hardware configuration for proper device settings:

- **Mac/CPU**: `+hardware=macbook` or `+hardware=mac_parallel`
- **AWS/GPU**: `+hardware=aws`
- **Multi-GPU**: `+hardware=pufferbox`

## Time Estimates

| Config | Parameters | Est. Time per Run | Total Sweep Time |
|--------|------------|-------------------|------------------|
| quick_sweep | 3 | 5-10 mins | ~30 mins |
| demo_sweep | 4 | 15-30 mins | 2-4 hours |
| protein_simple | 4 | 30-60 mins | 4-8 hours |
| protein_complex | 10+ | 1-2 hours | 8+ hours |

## Best Practices

1. **Start small**: Use `quick_sweep` to test your setup
2. **Hardware matching**: Always specify correct hardware config
3. **Monitoring**: Enable WandB for real-time tracking
4. **Resource planning**: Ensure adequate compute for sweep duration
5. **Parameter selection**: Focus on parameters with highest impact first
