# Sweep Configurations

This directory contains hyperparameter sweep configurations for the Metta training system using the **Protein optimizer** (Gaussian Process-based optimization).

## 🚀 **Sweep Pipeline Overview**

The Metta sweep system automates hyperparameter optimization through these steps:

1. **Configuration** → Define parameter search space and training overrides
2. **Initialization** → Create WandB sweep with Protein optimizer
3. **Optimization Loop** → Protein suggests parameters → Train → Evaluate → Repeat
4. **Summary** → Analyze results and find best hyperparameters

```mermaid
graph LR
    A[Sweep Config] --> B[sweep_init.py]
    B --> C[WandB Sweep]
    C --> D[Protein Optimizer]
    D --> E[Parameter Suggestion]
    E --> F[Training Run]
    F --> G[Evaluation]
    G --> D
    G --> H[Summary Report]
```

## 📋 **Configuration Format**

### ✅ **Correct Format: Nested Structure (Required)**

```yaml
# @package _global_
# IMPORTANT: All trainer/env overrides MUST be nested under 'sweep' section

# Rollout control
rollout_count: 10      # Number of optimization iterations
num_samples: 1         # Training runs per iteration

sweep:
  # Parameter search space
  parameters:
    # Learning rate with log-normal distribution
    trainer.learning_rate:
      min: 0.00001
      max: 0.01
      mean: 0.001
      scale: 1
      distribution: log_normal

    # Batch size with power-of-2 constraint
    trainer.batch_size:
      min: 32
      max: 256
      mean: 64
      scale: 1
      distribution: uniform_pow2  # Ensures power-of-2 values

    # Discount factor with uniform distribution
    trainer.gamma:
      min: 0.95
      max: 0.999
      mean: 0.99
      scale: 1
      distribution: uniform

  # Optimizer settings
  metric: reward
  goal: maximize

  # ⚠️ CRITICAL: Trainer overrides MUST be nested here
  trainer:
    total_timesteps: 50000
    evaluate_interval: 5000
    checkpoint_interval: 10000
    minibatch_size: 32

  # ⚠️ CRITICAL: Environment overrides MUST be nested here
  env:
    game:
      max_steps: 64
      objects:
        altar:
          hp: 10
          cooldown: 5
```

### ❌ **Incorrect Format: Root-Level Overrides (Won't Work)**

```yaml
# DON'T DO THIS - Overrides at root level are ignored!
rollout_count: 10

sweep:
  parameters:
    trainer.learning_rate:
      distribution: log_normal
      # ...

# ❌ WRONG: These overrides won't be applied
trainer:
  total_timesteps: 50000

env:
  game:
    max_steps: 64
```

## 🎯 **End-to-End Example: 1-Minute Test Sweep**

### 1. **Create Config File** (`configs/sweep/my_test_sweep.yaml`)

```yaml
# @package _global_
# Quick test sweep - runs in ~1 minute

rollout_count: 3    # 3 optimization rounds
num_samples: 1      # 1 run per round

sweep:
  parameters:
    # Optimize 3 key parameters
    trainer.learning_rate:
      min: 0.0001
      max: 0.01
      mean: 0.001
      scale: 1
      distribution: log_normal

    trainer.batch_size:
      min: 32
      max: 128
      mean: 64
      scale: 1
      distribution: uniform_pow2

    trainer.gamma:
      min: 0.95
      max: 0.999
      mean: 0.99
      scale: 1
      distribution: uniform

  metric: reward
  goal: maximize

  # Training overrides for quick testing
  trainer:
    total_timesteps: 2048        # ~20 seconds per run
    evaluate_interval: 10
    checkpoint_interval: 20
    minibatch_size: 16
    num_steps: 32
    update_epochs: 1

  # Environment overrides
  env:
    game:
      max_steps: 32              # Short episodes
      objects:
        altar:
          hp: 5
          cooldown: 5
```

### 2. **Run the Sweep**

```bash
# For Mac/local testing
./devops/sweep.sh run=my_test ++sweep_params=sweep/my_test_sweep +hardware=macbook +user=axel trainer.num_workers=1

# For AWS/GPU
./devops/sweep.sh run=my_test ++sweep_params=sweep/my_test_sweep +hardware=aws
```

### 3. **Monitor Progress**

The sweep will output:
- WandB sweep URL for real-time monitoring
- Individual run directories in `train_dir/sweep/my_test/runs/`
- Protein suggestions in each run directory

### 4. **View Summary**

```bash
# After sweep completes (or during)
./summarize_sweep.sh my_test
```

Output example:
```
📊 SWEEP SUMMARY: my_test
=================================================================

📋 SWEEP CONFIGURATION:
=================================================================
Rollout count limit: 3
Samples per rollout: 1

Parameter space:
  trainer.learning_rate: log_normal [0.0001, 0.01]
  trainer.batch_size: uniform_pow2 [32, 128]
  trainer.gamma: uniform [0.95, 0.999]

Trainer overrides:
  total_timesteps: 2048
  evaluate_interval: 10
  ...

🎯 ROLLOUT RESULTS:
=================================================================
Rollout 0: 3 runs
  Best reward: 2.45 (run my_test.r.0)
  Parameters: lr=0.0023, batch=64, gamma=0.98

Rollout 1: 3 runs
  Best reward: 3.12 (run my_test.r.3)
  Parameters: lr=0.0045, batch=128, gamma=0.99

Rollout 2: 3 runs
  Best reward: 3.89 (run my_test.r.6)
  Parameters: lr=0.0031, batch=64, gamma=0.995

🏆 OVERALL BEST RUN: my_test.r.6
  Reward: 3.89
  Config: train_dir/sweep/my_test/runs/my_test.r.6/config.yaml
```

## 📊 **Available Sweep Configurations**

### Quick Testing (Minutes)
- **`protein_working.yaml`** - 1-minute e2e test (3 rollouts, 2K timesteps)
- **`protein_lightning.yaml`** - 9-minute sweep (3 rollouts, 5 timesteps)

### Development (Hours)
- **`protein_fast.yaml`** - 1-2 hour sweep (10 rollouts, 5K timesteps)
- **`protein_simple.yaml`** - 4-parameter optimization

### Production (Many Hours)
- **`full_protein.yaml`** - Comprehensive parameter search
- **`protein_complex.yaml`** - 10+ parameter optimization

### Environment-Specific
- **`pong.yaml`** - Optimized for Pong environment
- **`cogeval_sweep.yaml`** - Cognitive evaluation tasks

### Templates
- **`my_sweep.yaml`** - User customization template
- **`empty.yaml`** - Empty template for new sweeps

## 🔧 **Parameter Distribution Types**

| Distribution | Use Case | Example Values |
|-------------|----------|----------------|
| `uniform` | Linear range | 0.1, 0.2, 0.3, ... |
| `log_normal` | Learning rates | 1e-5, 1e-4, 1e-3, ... |
| `uniform_pow2` | Batch sizes | 32, 64, 128, 256 |
| `int_uniform` | Integer ranges | 1, 2, 3, 4, ... |

## ⏱️ **Time Estimates**

| Config | Parameters | Rollouts | Est. Time per Run | Total Time |
|--------|------------|----------|-------------------|------------|
| protein_working | 5 | 3 | ~20 secs | ~1 min |
| protein_lightning | 1 | 3 | ~3 mins | ~9 mins |
| protein_fast | 8 | 10 | ~10 mins | ~2 hours |
| protein_simple | 4 | 20 | ~30 mins | ~10 hours |
| full_protein | 10+ | 50 | ~1 hour | ~50 hours |

## 💡 **Best Practices**

1. **Always use nested structure** - Put trainer/env overrides under `sweep` section
2. **Use power-of-2 for batch parameters** - Prevents divisibility conflicts
3. **Start with few parameters** - Add more once baseline works
4. **Set appropriate rollout_count** - More rollouts = better optimization but longer runtime
5. **Match hardware to workload** - Use `+hardware=macbook` for testing, `+hardware=aws` for production
6. **Monitor with WandB** - Check sweep progress in real-time
7. **Use num_samples wisely** - More samples = better statistics but longer runtime

## 🚨 **Common Issues & Solutions**

### Issue: "Trainer overrides not applied"
**Solution**: Ensure trainer config is nested under `sweep.trainer`, not at root level

### Issue: "Batch size not divisible by minibatch_size"
**Solution**: Use `uniform_pow2` distribution for batch parameters

### Issue: "Sweep runs forever"
**Solution**: Set `rollout_count` to limit optimization iterations

### Issue: "Can't find sweep summary"
**Solution**: Use exact sweep name from `run=` parameter in `./summarize_sweep.sh`

### Issue: "Out of memory"
**Solution**: Reduce batch_size range or use smaller minibatch_size

## 🔄 **Migration from CARBS**

If you have old CARBS configs with `${ss:...}` syntax:

| CARBS | Protein |
|-------|---------|
| `${ss:log, 1e-5, 1e-2}` | `distribution: log_normal, min: 0.00001, max: 0.01` |
| `${ss:pow2, 32, 256}` | `distribution: uniform_pow2, min: 32, max: 256` |
| `${ss:int, 1, 10}` | `distribution: int_uniform, min: 1, max: 10` |
| `${ss:logit, 0.1, 0.9}` | `distribution: uniform, min: 0.1, max: 0.9` |

Remember to:
1. Replace all `${ss:...}` with proper parameter definitions
2. Nest all overrides under the `sweep` section
3. Add `rollout_count` to limit iterations

## 🛠️ **Advanced Usage**

### Custom Metrics
```yaml
sweep:
  metric: custom_metric  # Your custom logged metric
  goal: minimize         # or maximize
```

### Multi-GPU Sweeps
```bash
./devops/sweep.sh run=gpu_sweep ++sweep_params=sweep/full_protein +hardware=pufferbox
```

### Conditional Parameters
```yaml
sweep:
  parameters:
    trainer.use_lstm:
      values: [true, false]

    # Only used when use_lstm is true
    trainer.lstm_hidden_size:
      min: 64
      max: 512
      distribution: uniform_pow2
```

## 📚 **Further Reading**

- [Protein Optimizer Documentation](https://github.com/uber-research/protein)
- [WandB Sweeps Guide](https://docs.wandb.ai/guides/sweeps)
- [Hydra Configuration](https://hydra.cc/docs/intro/)
