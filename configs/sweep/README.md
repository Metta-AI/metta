# Sweep Configurations

This directory contains hyperparameter sweep configurations for the Metta training system.

## Available Configurations

### Quick Testing & Development
- **`quick_sweep.yaml`** ⚡ - Ultra-fast sweep designed to complete in ~30 minutes
  - 3 parameters: learning_rate, gamma, batch_size
  - 50K timesteps per run (~5-10 mins each)
  - Ideal for: Testing sweep infrastructure, rapid iteration, debugging
  - Usage: `./devops/sweep.sh run=test ++sweep_params=sweep/quick_sweep +hardware=macbook`

- **`demo_sweep.yaml`** 🎯 - Demo configuration for testing
  - 4 parameters: learning_rate, gamma, batch_size, clip_coef
  - Standard timesteps
  - Usage: `./devops/sweep.sh run=demo ++sweep_params=sweep/demo_sweep +hardware=macbook`

- **`fast.yaml`** 🏃 - Fast convergence configuration
  - Multiple parameters with focused ranges
  - Low total timesteps (1e3-1e4)

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

Sweep configs use the Protein format with nested parameter definitions:

```yaml
# @package _global_
sweep:
  trainer:
    optimizer:
      learning_rate:
        distribution: log_normal
        min: 1e-5
        max: 1e-2
        mean: 1e-3
        scale: 1.0

    batch_size:
      distribution: int_uniform
      min: 32
      max: 128
      mean: 64
      scale: 1.0

# Override training settings
trainer:
  total_timesteps: 50000
  evaluate_interval: 5000
```

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
