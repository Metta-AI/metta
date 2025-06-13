# Metta Protein Optimization & Sweeps

This directory contains the **Protein optimizer** implementation and documentation for hyperparameter optimization sweeps using **Protein** integrated with **Weights & Biases (WandB)**.

## Overview

The Metta sweep system automatically handles the **WandB ↔ Protein optimization loop**, where:
1. **WandB** manages experiment tracking and distributed coordination
2. **Protein** provides intelligent Bayesian optimization suggestions
3. **Protein suggestions override WandB's parameters** for superior optimization

## Core Components

### Files in this Directory
- **`metta_protein.py`** - Main `MettaProtein` class integrating Protein with Metta configs
- **`wandb_protein.py`** - Base `WandbProtein` class handling WandB integration
- **`__init__.py`** - Package initialization

### Related Files
- **`../../tools/sweep_init.py`** - Sweep initialization and Protein parameter override
- **`../../devops/sweep.sh`** - Main sweep execution script
- **`../../configs/sweep/`** - Sweep configuration files

## Quick Start

### Mac/CPU Training
For Mac or CPU-only environments, always include the hardware configuration:
```bash
# Demo sweep on Mac
./devops/sweep.sh run=demo_sweep_test ++sweep_params=sweep/demo_sweep +hardware=macbook +user=<username>

# Optimized for Mac with custom worker count
./devops/sweep.sh run=trial_sweep ++sweep_params=sweep/demo_sweep +hardware=macbook +user=<username> trainer.num_workers=1
```

### GPU/Cloud Training
```bash
# Simple 4-parameter optimization
./devops/sweep.sh run=my_protein_simple ++sweep_params=sweep/protein_simple +hardware=aws

# Complex 10-parameter optimization
./devops/sweep.sh run=my_protein_complex ++sweep_params=sweep/protein_complex +hardware=aws
```

## How It Works

### Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   WandB Sweep   │───▶│  Protein Opt    │───▶│   Training      │
│  (Coordination) │    │ (Smart Params)  │    │   (Results)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                                              │
         └──────────────────────────────────────────────┘
                        Feedback Loop
```

### Workflow
1. **`./devops/sweep.sh`** - Main entry point with retry logic
2. **`./devops/sweep_rollout.sh`** - Single sweep execution
3. **`./tools/sweep_init.py`** - Creates WandB sweep and runs Protein optimization
4. **`./devops/train.sh`** - Distributed training with Protein's parameters
5. **`./tools/sweep_eval.py`** - Evaluation and metric reporting

### Key Innovation: Parameter Override
```python
# In tools/sweep_init.py
wandb.agent(sweep_id, function=init_run, count=1)

def init_run():
    # WandB suggests parameters, but we override with Protein
    protein = MettaProtein(cfg, wandb_run)
    suggestion, _ = protein.suggest()  # Protein's intelligent suggestion
    apply_protein_suggestion(train_cfg, suggestion)  # Override WandB params
```

## Protein Implementation Details

### MettaProtein Class
```python
from metta.rl.protein_opt.metta_protein import MettaProtein

# Initialize with config and WandB run
protein = MettaProtein(cfg, wandb_run)

# Get intelligent parameter suggestions
suggestion, metadata = protein.suggest()

# Report results for learning
protein.observe(suggestion_uuid, reward, context)
```

### WandbProtein Base Class
The `WandbProtein` class handles the critical WandB integration:
```python
# Key method that overrides WandB's parameters
wandb_run.config.update(wandb_config, allow_val_change=True)
```

This ensures Protein's suggestions take precedence over WandB's naive sampling.

## Configuration Structure

### Standard Sweep Config Format
```yaml
# configs/sweep/my_sweep.yaml

# Basic WandB sweep structure (for compatibility)
method: bayes
metric:
  name: reward
  goal: maximize

# Protein-specific configuration (the real optimizer)
sweep:
  metric: reward
  goal: maximize
  num_random_samples: 3  # Initial exploration

  parameters:
    trainer.learning_rate:
      min: 1e-5
      max: 1e-2
      distribution: log_normal

    trainer.batch_size:
      min: 32
      max: 128
      distribution: int_uniform

    trainer.gamma:
      min: 0.9
      max: 0.999
      distribution: uniform
```

### Parameter Distributions

| Distribution | Use Case | Example |
|-------------|----------|---------|
| `uniform` | Linear range | `gamma: 0.9 to 0.999` |
| `log_normal` | Exponential range | `learning_rate: 1e-5 to 1e-2` |
| `int_uniform` | Integer range | `batch_size: 32 to 128` |

## Available Configurations

### Demo & Testing
- **`configs/sweep/demo_sweep.yaml`** - Quick 3-parameter demo for testing
- **`configs/sweep/fast.yaml`** - Minimal configuration for rapid iteration
- **`configs/sweep/empty.yaml`** - Template for new sweeps

### Production Sweeps
- **`configs/sweep/protein_simple.yaml`** - 4 core hyperparameters
- **`configs/sweep/protein_complex.yaml`** - 10 parameters, ~10^23 combinations
- **`configs/sweep/pong.yaml`** - Full Pong environment optimization
- **`configs/sweep/cogeval_sweep.yaml`** - Cognitive evaluation tasks

### Environment-Specific
- **`configs/sweep/full.yaml`** - Comprehensive parameter space
- **`configs/sweep/my_sweep.yaml`** - User customization template

## Creating Custom Sweeps

### 1. Create Configuration File
```yaml
# configs/sweep/my_custom_sweep.yaml
sweep:
  metric: reward
  goal: maximize
  num_random_samples: 5

  parameters:
    # Learning rate with log-normal distribution
    trainer.learning_rate:
      min: 1e-5
      max: 1e-2
      mean: 1e-3
      scale: 1
      distribution: log_normal

    # Batch size with discrete values
    trainer.batch_size:
      min: 32
      max: 256
      distribution: int_uniform

    # Model architecture parameters
    model.hidden_size:
      min: 64
      max: 512
      distribution: int_uniform

    # Environment-specific parameters
    env.difficulty:
      min: 0.1
      max: 1.0
      distribution: uniform
```

### 2. Run Your Sweep
```bash
# Mac/CPU training
./devops/sweep.sh run=my_experiment ++sweep_params=sweep/my_custom_sweep +hardware=macbook +user=<username>

# GPU/Cloud training
./devops/sweep.sh run=my_experiment ++sweep_params=sweep/my_custom_sweep +hardware=aws
```

## Parameter Guidelines

### Learning Rate
```yaml
trainer.learning_rate:
  min: 1e-5    # Conservative minimum
  max: 1e-2    # Aggressive maximum
  distribution: log_normal  # Essential for learning rates
```

### Batch Size
```yaml
trainer.batch_size:
  min: 32      # Memory constraints
  max: 256     # Hardware limits
  distribution: int_uniform
```

### PPO-Specific Parameters
```yaml
trainer.gamma:         # Discount factor
  min: 0.9
  max: 0.999

trainer.clip_param:    # PPO clipping
  min: 0.1
  max: 0.3

trainer.entropy_coeff: # Exploration
  min: 0.001
  max: 0.1
  distribution: log_normal
```

## Monitoring & Results

### Files Generated
```
train_dir/sweep/{sweep_name}/
├── config.yaml                 # Sweep metadata
├── runs/
│   └── {run_id}/
│       ├── protein_suggestion.yaml    # Protein's parameters
│       ├── protein_state.yaml         # Optimization state
│       ├── train_config_overrides.yaml # Final parameters
│       └── dist_{dist_id}.yaml        # Distributed config
```

### WandB Dashboard
- **Project**: `metta` (default)
- **Tags**: `sweep_id:{id}`, `sweep_name:{name}`
- **Metrics**: Tracked automatically during training
- **Parameters**: Protein suggestions visible in run config

## Advanced Usage

### Multi-Node Sweeps
```bash
# Node 0 (master)
NODE_INDEX=0 NUM_NODES=2 ./devops/sweep.sh run=distributed_sweep ++sweep_params=sweep/protein_complex +hardware=aws

# Node 1 (worker)
NODE_INDEX=1 NUM_NODES=2 ./devops/sweep.sh run=distributed_sweep ++sweep_params=sweep/protein_complex +hardware=aws
```

### Hardware Configurations
Choose the appropriate hardware configuration for your environment:

```bash
# Mac/CPU configurations
+hardware=macbook        # Single-core CPU training (for testing/development)
+hardware=mac_parallel   # Multi-core CPU training (better performance)
+hardware=mac_serial     # Minimal CPU training (most compatible)

# GPU/Cloud configurations
+hardware=aws           # AWS GPU instances with CUDA
+hardware=pufferbox     # Multi-GPU distributed training
+hardware=github        # CI/CD environments (minimal config)
```

**Important**: Always specify hardware configuration to avoid CUDA/NCCL errors on non-GPU systems.

### Environment Variables
```bash
export WANDB_PROJECT=my_project    # Custom WandB project
export WANDB_ENTITY=my_team        # Custom WandB entity
export NUM_GPUS=4                  # Multi-GPU training
export NUM_CPUS=16                 # CPU allocation
```

### Continuous Sweeps
The sweep system includes automatic retry logic:
- **Max consecutive failures**: 3
- **Retry delay**: 5 seconds
- **Auto-recovery**: Resumes from last successful state

## Testing & Validation

### Test Suite
The Protein optimization system includes comprehensive tests:

```bash
# Run all protein/sweep tests
python -m pytest tests/rl/ -k "protein or sweep" -v

# Run specific test suites
python -m pytest tests/rl/test_protein_e2e.py -v                    # End-to-end tests
python -m pytest tests/rl/test_protein_comprehensive.py -v          # Core functionality
python -m pytest tests/rl/test_sweep_integration.py -v              # Mac compatibility
```

### Test Coverage
- **End-to-End Workflow**: Complete sweep config → parameter optimization flow
- **WandB Integration**: Parameter override and result recording
- **Hardware Compatibility**: Mac CPU-only and GPU configurations
- **Config Validation**: Sweep configuration structure and parameter ranges
- **Error Handling**: Failure scenarios and recovery

### Validation Commands
```bash
# Validate sweep config structure
./tools/sweep_init.py --validate configs/sweep/demo_sweep.yaml

# Test Mac hardware compatibility
./devops/sweep.sh run=test_mac ++sweep_params=sweep/demo_sweep +hardware=macbook trainer.total_timesteps=100

# Check parameter override functionality
grep "protein" train_dir/test_mac/*/protein_suggestion.yaml
```

## Development & Debugging

### Adding New Parameter Types
To support new parameter distributions, modify:
1. **`metta_protein.py`** - Add parameter space handling
2. **`tools/sweep_init.py`** - Update `_convert_protein_params_to_wandb()`

### Custom Protein Variants
Extend `WandbProtein` for specialized optimizers:
```python
from metta.rl.protein_opt.wandb_protein import WandbProtein

class CustomProtein(WandbProtein):
    def suggest(self):
        # Your custom optimization logic
        return suggestion, metadata
```

### Debug Commands
```bash
# Check sweep status
ls -la train_dir/sweep/{sweep_name}/

# View protein suggestions
cat train_dir/sweep/{sweep_name}/runs/{run_id}/protein_suggestion.yaml

# Monitor training logs
tail -f train_dir/sweep/{sweep_name}/runs/{run_id}/train.log

# Check Protein state
cat train_dir/sweep/{sweep_name}/runs/{run_id}/protein_state.yaml
```

## Troubleshooting

### Common Issues

1. **Empty metrics in WandB**
   - Verify training is actually running (check logs)
   - Ensure `trainer.update_epochs` matches your distributed setup

2. **Protein not optimizing**
   - Check `protein_suggestion.yaml` files are being generated
   - Verify parameters are being applied in `train_config_overrides.yaml`
   - Ensure WandB run has proper tags for sweep tracking

3. **Memory issues with large batch sizes**
   - Reduce `trainer.batch_size` maximum in your sweep config
   - Increase `NUM_CPUS` environment variable

4. **Parameter override not working**
   - Check `WandbProtein.update_wandb_config()` is being called
   - Verify `allow_val_change=True` in WandB config update

### Best Practices

#### Parameter Selection
- **Start small**: Begin with 3-4 parameters
- **Log-normal for rates**: Always use log-normal for learning rates
- **Reasonable ranges**: Don't make ranges too wide initially
- **Domain knowledge**: Include parameters you suspect matter most

#### Optimization Strategy
- **Exploration phase**: Set `num_random_samples: 5-10` initially
- **Exploitation**: Let Protein take over after random sampling
- **Iteration**: Run multiple sweeps, narrowing ranges based on results

#### Resource Management
- **Batch size**: Constrain by available GPU memory
- **Parallel runs**: Don't exceed your compute capacity
- **Storage**: Monitor disk usage in `train_dir/sweep/`

## Integration with Other Systems

### SkyPilot Cloud Training
```bash
# Launch cloud sweep (see devops/skypilot/README.md)
sky launch --cluster my-sweep-cluster sweep_job.yaml --env SWEEP_CONFIG=configs/sweep/protein_complex.yaml
```

### Custom Evaluation Metrics
Modify `tools/sweep_eval.py` to include custom metrics:
```python
# Add your custom metrics
custom_metrics = {
    "custom_score": compute_custom_score(),
    "efficiency": compute_efficiency_metric()
}
wandb.log(custom_metrics)
```

## Related Documentation

- **[Main README](../../../README.md)** - Project overview
- **[Devops Documentation](../../../devops/README.md)** - Training pipeline
- **[Protein Paper](https://arxiv.org/abs/2308.00352)** - Algorithm details
- **[WandB Sweeps Documentation](https://docs.wandb.ai/guides/sweeps)** - WandB integration

---

**Note**: This Protein optimization system replaces traditional random/grid search with intelligent Bayesian optimization while maintaining full compatibility with WandB's tracking and coordination features.
