# Metta Clean API (No Hydra)

A clean, torch.rl-style API for using Metta as a library without Hydra configuration.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import metta_api as metta

# Create components with sensible defaults
env_config = metta.create_env(num_agents=4)
vecenv = metta.make_vecenv(env_config=env_config, num_envs=32)
policy = metta.make_agent(obs_space=obs_space, ...)
optimizer = metta.make_optimizer(policy.parameters())
loss_module = metta.make_loss_module(policy=policy)

# Standard PyTorch training loop
loss = loss_module(minibatch=minibatch, ...)
loss.backward()
optimizer.step()
```

## Simple Training Script

```bash
# Train with defaults
python run.py

# Train with options
python run.py train --run my_experiment --timesteps 1000000

# Evaluate
python run.py sim --run my_experiment --policy-uri file://./checkpoints/policy.pt
```

**Note:** Import `metta_api` instead of `metta` directly due to the package structure. The `metta_api` module provides all the clean API functions you need.

## API Reference

### Factory Functions

- `metta.make_agent()` - Create agent instances directly
- `metta.make_vecenv()` - Create vectorized environments
- `metta.make_optimizer()` - Create optimizers
- `metta.make_experience_buffer()` - Create experience buffers
- `metta.make_loss_module()` - Create loss modules (torch.nn.Module style)

### Configuration Helpers

- `metta.create_env()` - Create environment config
- `metta.create_agent()` - Create agent config
- `metta.create_trainer()` - Create trainer config
- `metta.build_runtime_config()` - Build runtime configuration
- `metta.setup_metta_environment()` - Setup Metta environment
- `metta.get_logger()` - Get configured logger

### Functional Training

From `metta.rl.functional_trainer`:
- `perform_rollout_step()` - Single rollout step
- `compute_initial_advantages()` - Compute GAE advantages
- `process_rollout_infos()` - Process rollout statistics

### Loss Modules

From `metta.rl.objectives`:
- `ClipPPOLoss` - PPO loss module (torch.nn.Module)

## Example: Complete Training Loop

See `examples/clean_api_example.py` for a complete example, or the simplified `run.py`.

```python
# Training loop
while agent_step < total_timesteps:
    # Rollout
    experience.reset_for_rollout()
    while not experience.ready_for_training:
        num_steps, info, _ = perform_rollout_step(policy, vecenv, experience, device, timer)
        agent_step += num_steps

    # Train
    advantages = compute_initial_advantages(experience, gamma, gae_lambda, 1.0, 1.0, device)

    for minibatch in experience.sample_minibatches():
        loss = loss_module(minibatch=minibatch, ...)
        loss.backward()
        optimizer.step()
```

## Key Benefits

1. **No Hydra Required** - Direct instantiation of all components
2. **torch.rl Style** - Loss modules inherit from nn.Module
3. **Modular Design** - Small, composable functions
4. **Clean API** - Simple factory functions with sensible defaults
5. **PyTorch Native** - Standard PyTorch training loops

The API maintains full compatibility with existing Metta functionality while providing a much cleaner interface for users who want to use Metta as a library.
