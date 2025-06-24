# Quick Start: Functional Training in Metta

This guide shows you how to use the new modular, functional training approach in Metta.

## Installation

First, make sure you have the environment set up:
```bash
cd /path/to/metta
uv venv
source .venv/bin/activate  # or .venv/Scripts/activate on Windows
uv pip install -e .
cd mettagrid && uv pip install -e . && cd ..
cd common && uv pip install -e . && cd ..
```

## 4 Ways to Train

### 1. Quick Start (Minimal Code)

```python
from metta.agent import create_agent
from metta.agent.policy_store import MemoryPolicyStore
from metta.rl.functional_trainer import functional_training_loop
from metta.train.train_config import small_fast_config
from mettagrid import create_env
from mettagrid.curriculum import SimpleCurriculum

# Use preset config
config = small_fast_config()

# Create minimal components
env = create_env(width=10, height=10)
agent = create_agent("simple_cnn", env.observation_space, env.action_space)
curriculum = SimpleCurriculum(base_env_config={"width": 10, "height": 10})
policy_store = MemoryPolicyStore()

# Run training
state = functional_training_loop(
    config=config.trainer,
    ppo_config=config.ppo,
    policy=agent,
    curriculum=curriculum,
    policy_store=policy_store,
)
```

### 2. Structured Configs (Recommended)

```python
from metta.rl.configs import PPOConfig, TrainerConfig
from metta.train.train_config import TrainingConfig

# Create structured config
config = TrainingConfig(
    experiment_name="my_experiment",
    trainer=TrainerConfig(
        total_timesteps=100_000,
        batch_size=2048,
        device="cuda",
    ),
    ppo=PPOConfig(
        clip_coef=0.2,
        ent_coef=0.01,
    ),
)

# Use config values throughout your code
# config.trainer.total_timesteps, config.ppo.clip_coef, etc.
```

### 3. Direct Component Usage (Maximum Control)

```python
from metta.rl.collectors import RolloutCollector
from metta.rl.optimizers import PPOOptimizer
from metta.rl.experience import Experience

# Create components
collector = RolloutCollector(vecenv, policy, experience, device)
ppo = PPOOptimizer(policy, optimizer, device, PPOConfig())

# Custom training loop
for epoch in range(1000):
    # Collect data
    stats, steps = collector.collect()

    # Update policy
    losses = ppo.update(experience)

    # Your custom logic here
    if epoch % 100 == 0:
        save_checkpoint()
```

### 4. Custom Training Steps & Losses

```python
# Define custom loss
def curiosity_loss(policy, obs, actions, rewards, values, advantages):
    """Your custom loss function."""
    return loss_tensor

# Define custom training step
def my_training_step(state, components, config, experience, custom_losses=None):
    """Custom training logic."""
    metrics = default_training_step(state, components, config, experience, custom_losses)

    # Add your logic
    if should_stop_early(metrics):
        state.should_stop = True

    return metrics

# Use in training
state = functional_training_loop(
    config=trainer_config,
    ppo_config=ppo_config,
    policy=agent,
    curriculum=curriculum,
    policy_store=policy_store,
    step_fn=my_training_step,
    custom_losses=[curiosity_loss],
)
```

## Running the Examples

1. **Basic training example:**
   ```bash
   python examples/run_functional_training.py
   ```

2. **With custom losses:**
   ```bash
   python examples/functional_training_example.py
   ```

3. **Direct component usage:**
   ```bash
   python examples/direct_training_with_components.py
   ```

## Key Concepts

### Structured Configs
- Replace YAML with type-safe dataclasses
- IDE autocomplete and validation
- Easy to modify programmatically

### Functional Training Loop
- Explicit control flow
- Easy to modify or extend
- Clear separation of concerns

### Modular Components
- `RolloutCollector`: Environment interaction
- `PPOOptimizer`: Policy optimization
- `PolicyEvaluator`: Evaluation
- `TrainingCheckpointer`: Saving/loading
- `StatsLogger`: Metrics logging

### Custom Losses
- Add any PyTorch loss function
- Access to policy, observations, actions, etc.
- Automatically integrated into PPO update

## Migrating from Old Code

If you have old YAML configs:
```python
# Old way
yaml_config = load_yaml("config.yaml")

# New way
from metta.train.train_config import TrainingConfig
config = TrainingConfig.from_dict(yaml_config)
```

If you're using the old trainer:
```python
# Old way
trainer = MettaTrainer(cfg, wandb_run, policy_store, sim_config, stats_client)
trainer.train()

# New way (functional)
state = functional_training_loop(
    config=config.trainer,
    ppo_config=config.ppo,
    policy=agent,
    curriculum=curriculum,
    policy_store=policy_store,
)
```

## Tips

1. **Start simple**: Use `small_fast_config()` for testing
2. **Use structured configs**: Better than dictionaries
3. **Leverage presets**: `small_fast_config()`, `medium_config()`, `large_distributed_config()`
4. **Add custom losses gradually**: Test with default training first
5. **Monitor training**: Use wandb integration or custom logging

## Troubleshooting

- **Import errors**: Make sure mettagrid and common are installed
- **CUDA errors**: Set `device="cpu"` in configs if no GPU
- **Memory issues**: Reduce `batch_size` and `num_envs`
- **Slow training**: Use `compile=True` in trainer config

## Next Steps

- Explore the [component documentation](../metta/rl/README.md)
- Read about [custom loss functions](functional_training_example.py)
- Check out [distributed training](../metta/train/train_config.py) configs
