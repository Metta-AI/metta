# Functional Training in Metta

This document explains the new functional training approach in Metta, which provides a simpler, more explicit alternative to the existing Hydra-based training system.

## Overview

The functional training approach allows you to:
- Create all objects explicitly without framework magic
- Use simple Pydantic configs instead of complex YAML hierarchies
- Run training with a straightforward while loop
- Bypass Hydra configuration complexity
- Have full control over the training process

## Quick Start

Run the demo to see functional training in action:

```bash
python demo.py
```

This will train a simple agent for 10,000 steps, saving checkpoints every 5 epochs.

## Core Components

### 1. Functional Training Functions

The core training logic has been extracted into two main functions in `metta/rl/functional_trainer.py`:

- `rollout()` - Collects experience from the environment
- `train_ppo()` - Updates the policy using PPO algorithm

### 2. Pydantic Configurations

Instead of complex YAML files, use simple Pydantic models:

```python
from pydantic import BaseModel

class PPOConfig(BaseModel):
    gamma: float = 0.977
    gae_lambda: float = 0.916
    clip_coef: float = 0.1
    # ... etc

# Create config directly
ppo_config = PPOConfig(gamma=0.99, clip_coef=0.2)

# Or load from YAML
ppo_config = load_yaml_config("config.yaml", PPOConfig)
```

### 3. Direct Object Creation

Create all components explicitly:

```python
# Create environment
vecenv = make_vecenv(curriculum, "serial", num_envs=4)

# Create agent
policy = MettaAgent(...)
policy.activate_actions(action_names, max_params, device)

# Create experience buffer
experience = Experience(
    total_agents=num_agents,
    batch_size=batch_size,
    obs_space=obs_space,
    atn_space=atn_space,
    device=device,
)

# Create optimizer
optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
```

### 4. Simple Training Loop

The training loop is just a while loop:

```python
while agent_step < total_timesteps:
    # Collect experience
    agent_step, stats = rollout(
        policy, vecenv, experience, device, agent_step, timer
    )

    # Update policy
    epoch = train_ppo(
        policy=policy,
        optimizer=optimizer,
        experience=experience,
        device=device,
        losses=losses,
        epoch=epoch,
        # ... PPO parameters
    )

    # Log metrics and save checkpoints
    print(f"Epoch {epoch}: {steps_per_sec} SPS")
```

## Comparison with Hydra-based Training

| Aspect | Hydra Training | Functional Training |
|--------|---------------|-------------------|
| Configuration | Complex YAML hierarchy | Simple Pydantic models |
| Object Creation | Magic instantiation | Explicit creation |
| Training Loop | Hidden in Trainer class | Visible while loop |
| Flexibility | Limited by framework | Full control |
| Debugging | Harder | Straightforward |
| Learning Curve | Steep | Gentle |

## Use Cases

The functional approach is ideal for:
- Research requiring custom training loops
- Debugging and understanding the training process
- Integration with other frameworks
- Educational purposes
- Rapid prototyping

## Migration Guide

If you have existing Hydra configs, you can gradually migrate:

1. Keep using `tools/train.py` for production training
2. Use `demo.py` as a template for experiments
3. Load existing YAML configs into Pydantic models
4. Gradually replace Hydra components

## Advanced Usage

### Custom Training Step

```python
def my_training_loop(policy, vecenv, experience, optimizer):
    for epoch in range(1000):
        # Custom rollout logic
        agent_step, stats = custom_rollout(...)

        # Custom advantage computation
        advantages = compute_custom_advantages(...)

        # Custom PPO update
        losses = custom_ppo_update(...)

        # Your metrics
        log_custom_metrics(...)
```

### Integration with Other Libraries

Since everything is explicit, it's easy to integrate with:
- TensorBoard
- Custom logging systems
- Different RL algorithms
- External optimizers
- Custom environments

## Performance

The functional approach has the same performance as the class-based approach:
- Uses the same optimized C++/CUDA kernels
- Same parallelization capabilities
- Same memory efficiency
- Just more transparent

## Next Steps

1. Run `demo.py` to see it in action
2. Modify the demo for your use case
3. Check `metta/rl/functional_trainer.py` for implementation details
4. Create your own training loops

The functional approach makes Metta's training process transparent and hackable while maintaining all the performance benefits of the original implementation.
