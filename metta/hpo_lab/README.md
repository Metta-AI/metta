# HPO Lab - Hyperparameter Optimization Prototyping Environment

A lightweight, fast-iterating environment for prototyping and benchmarking hyperparameter optimization (HPO) algorithms using standard RL benchmarks.

## Overview

The HPO Lab provides:
- **Fast iteration**: LunarLander training in 3-5 minutes (vs hours for MettaGrid)
- **Battle-tested implementations**: Uses StableBaselines3 for reliable RL
- **Full integration**: Works with Metta's Ray/Optuna sweep infrastructure
- **Cloud-ready**: Automatic support for WandB, S3, and Skypilot dispatch

## Quick Start

### Installation

The HPO Lab is included in the main Metta repository. Make sure you have the latest dependencies:

```bash
# Install/update dependencies
metta install

# Or specifically update SB3
uv add stable-baselines3@latest
```

### Basic Training

Test that everything works with a simple training run:

```python
# Quick test from Python
from metta.hpo_lab.recipes.lunarlander import train
metrics = train(total_timesteps=500000)  # ~2-3 minutes
print(f"Final reward: {metrics['final_mean_reward']:.2f}")
```

Or from the command line:

```bash
# Run training via CLI
uv run python -c "from metta.hpo_lab.recipes.lunarlander import train; print(train())"
```

### Running HPO Sweeps

#### Local Testing (Quick)

Run a small sweep locally to test your setup:

```bash
# Mini sweep with 10 trials (~30 minutes total)
uv run ./tools/run.py metta.hpo_lab.recipes.lunarlander.mini_sweep \
    sweep_name="test_local"
```

#### Cloud Sweeps (Production)

Run full sweeps on the cloud with GPU acceleration:

```bash
# Launch on cloud with GPUs (100 trials)
uv run ./tools/run.py metta.hpo_lab.recipes.lunarlander.ray_sweep \
    sweep_name="ak.lunarlander.v1" \
    -- gpus=8 nodes=2

# Monitor progress in WandB
# Results will appear under the sweep name
```

## Architecture

### Components

```
metta/hpo_lab/
├── trainers/
│   └── sb3_trainer.py       # StableBaselines3 wrapper
├── recipes/
│   └── lunarlander.py       # LunarLander benchmark recipe
└── README.md                # This file
```

### SB3Trainer

The `SB3Trainer` class wraps StableBaselines3 algorithms for sweep compatibility:

```python
from metta.hpo_lab.trainers import SB3Trainer

trainer = SB3Trainer(
    env_id="LunarLander-v3",
    algorithm="PPO",  # or "A2C", "SAC"
    total_timesteps=1_000_000,
    learning_rate=3e-4,
    n_steps=2048,
    # ... other hyperparameters
)

metrics = trainer.train()
# Returns: {
#   'final_mean_reward': 250.3,
#   'training_time': 287.4,
#   'success_rate': 1.0,
#   ...
# }
```

### Recipe Structure

Recipes provide sweep-compatible training functions:

```python
def train(**hyperparameters):
    """Training function called by sweep infrastructure."""
    trainer = SB3Trainer(**hyperparameters)
    return trainer.train()  # Returns metrics dict

def ray_sweep(sweep_name: str):
    """Sweep configuration for HPO experiments."""
    return make_sweep(
        name=sweep_name,
        parameters=[...],  # Hyperparameter search space
        objective="final_mean_reward",  # Metric to optimize
        ...
    )
```

## Hyperparameter Search Space

The default LunarLander sweep explores:

| Parameter | Range | Type | Description |
|-----------|-------|------|-------------|
| learning_rate | [1e-5, 1e-2] | log-uniform | Learning rate for optimizer |
| n_steps | [512, 4096] | categorical | Steps per environment per update |
| batch_size | [32, 256] | categorical | Minibatch size for gradient updates |
| n_epochs | [3, 20] | integer | Number of epochs per update |
| gamma | [0.95, 0.999] | uniform | Discount factor |
| gae_lambda | [0.9, 0.99] | uniform | GAE lambda for advantage estimation |
| clip_range | [0.1, 0.3] | uniform | PPO clipping parameter |
| ent_coef | [1e-6, 1e-2] | log-uniform | Entropy coefficient |
| net_arch | Various | categorical | Network architecture ([64,64], [128,128], etc.) |

## Performance Expectations

### LunarLander-v3
- **Target**: Average reward > 200 (solved)
- **Training time**: 3-5 minutes on GPU, 8-10 minutes on CPU
- **Timesteps to solve**: 300k-500k with good hyperparameters
- **Success rate**: >95% with optimized hyperparameters

### Sweep Performance
- **Single trial**: 3-5 minutes
- **10 trials (sequential)**: ~1 hour
- **100 trials (8 parallel)**: ~10 hours
- **1000 trials (16 parallel)**: ~2 days

## Advanced Usage

### Custom Environments

Add support for new environments:

```python
# In a new recipe file
def train_cartpole(**kwargs):
    return SB3Trainer(
        env_id="CartPole-v1",
        total_timesteps=50000,  # CartPole is easier
        **kwargs
    ).train()
```

### Different Algorithms

The trainer supports PPO, A2C, and SAC:

```python
# Continuous control with SAC
trainer = SB3Trainer(
    env_id="Pendulum-v1",  # Continuous action space
    algorithm="SAC",
    total_timesteps=100000,
)
```

### Custom Networks

Modify network architecture via `policy_kwargs`:

```python
trainer = SB3Trainer(
    policy_kwargs={
        "net_arch": [400, 300],  # Larger network
        "activation_fn": torch.nn.ReLU,
    }
)
```

## Integration with Metta Infrastructure

The HPO Lab automatically integrates with:

- **WandB**: Metrics logged automatically if configured
- **S3/Cloud Storage**: Model checkpoints can be saved/loaded
- **Skypilot**: Cloud dispatch works out of the box
- **Ray**: Distributed sweeps with resource management
- **Optuna**: Bayesian optimization for efficient search

## Troubleshooting

### Common Issues

1. **Import errors**: Run `metta install` to update dependencies
2. **CUDA errors**: Set `device="cpu"` if GPU unavailable
3. **Slow training**: Reduce `n_envs` or use smaller networks
4. **Out of memory**: Reduce `batch_size` or `n_steps`

### Debugging Tips

```python
# Verbose training to see progress
trainer = SB3Trainer(verbose=1)

# Test with shorter training
metrics = train(total_timesteps=10000)  # Quick test

# Check device usage
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Future Extensions

Planned additions:
- Atari environments (Pong, Breakout)
- MuJoCo continuous control (HalfCheetah, Humanoid)
- Custom network architectures (Transformers, xLSTM)
- Advanced HPO algorithms (Population-Based Training, ASHA)
- Multi-objective optimization support

## Contributing

To add new features:

1. Create new trainers in `trainers/` for different algorithms
2. Add recipes in `recipes/` for new environments
3. Ensure sweep compatibility (return metrics dict)
4. Add tests and documentation

## References

- [StableBaselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Environments](https://gymnasium.farama.org/)
- [Ray Tune HPO](https://docs.ray.io/en/latest/tune/index.html)
- [Optuna Optimization](https://optuna.org/)