# Metta API Documentation

The Metta API provides a clean, programmatic interface to the Metta training framework without requiring Hydra configuration files. This allows you to use Metta components directly in your Python code while maintaining full compatibility with the existing infrastructure.

## Quick Start

### Option 1: Using uv run (Recommended)

The simplest way to run API scripts is to add the uv shebang:

```python
#!/usr/bin/env -S uv run
```

This automatically handles environment setup and dependencies, similar to how scripts in the `tools/` folder work.

### Option 2: Manual Setup

If you prefer manual setup:

```bash
# Install dependencies
uv sync --inexact

# Run your script
python run.py
```

## Design Philosophy

The API provides:
- **Direct instantiation** of environments, agents, and training components
- **Pydantic configuration classes** for type-safe, validated settings
- **Exposed training loop components** giving you full visibility and control
- **Real training infrastructure** - not simplified wrappers, but the actual components used by MettaTrainer
- **Advanced features** - evaluation, monitoring, replay generation, all without Hydra

## Enhanced Features

The enhanced `run.py` example demonstrates professional-grade features:

### ✅ Core Training
- Full PPO training loop with experience replay
- Checkpointing and restoration
- Learning rate scheduling
- Gradient clipping and optimization

### ✅ Policy Evaluation
- Evaluate policies on multiple environments using `SimulationSuite`
- Track performance across different tasks
- Automatic score aggregation and logging

### ✅ Replay Generation
- Generate replay files for visualization in MetaScope
- Automatic URL generation for easy sharing
- Configurable replay intervals

### ✅ Memory & System Monitoring
- Track memory usage of training components
- Monitor CPU, GPU, and system resources
- Process-specific metrics
- Temperature monitoring (where available)

### ✅ Gradient Statistics
- Compute gradient mean, variance, and norm
- Track gradient behavior over training
- Configurable computation intervals

## Configuration Classes

Metta uses Pydantic models for configuration, providing validation and type safety:

### Why Use Structured Config Classes?

Using the structured Pydantic config classes provides several benefits:

1. **Type Safety**: IDEs can provide autocompletion and catch type errors at development time
2. **Validation**: Pydantic validates all values and provides clear error messages
3. **Documentation**: Each field has type annotations and docstrings explaining its purpose
4. **Defaults**: Smart defaults based on empirical testing and best practices
5. **Composition**: Config classes can be composed together cleanly

For example, instead of error-prone dictionaries:
```python
# Error-prone dictionary approach
config = {"ppo": {"clip_coef": "0.1"}}  # String instead of float!
```

Use typed configs:
```python
# Type-safe structured approach
config = TrainerConfig(
    ppo=PPOConfig(clip_coef=0.1),  # IDE catches type errors
)
```

### TrainerConfig
The main configuration class containing all training parameters:

```python
from metta.api import (
    TrainerConfig, PPOConfig, OptimizerConfig,
    CheckpointConfig, TorchProfilerConfig
)

# Option 1: Create with structured config classes (Recommended)
config = TrainerConfig(
    num_workers=4,
    total_timesteps=50_000_000,
    batch_size=524288,
    minibatch_size=16384,
    ppo=PPOConfig(
        clip_coef=0.2,
        ent_coef=0.01,
        gamma=0.99,
    ),
    optimizer=OptimizerConfig(
        type="adam",
        learning_rate=3e-4,
    ),
    checkpoint=CheckpointConfig(
        checkpoint_dir="./checkpoints",
        checkpoint_interval=100,
    ),
)

# Option 2: Use helper function with dict overrides (for compatibility)
from metta.api import create_default_trainer_config

config = create_default_trainer_config(
    num_workers=4,
    total_timesteps=50_000_000,
    batch_size=524288,
    ppo={"clip_coef": 0.2, "ent_coef": 0.01}
)
```

### PPOConfig
PPO algorithm hyperparameters:

```python
from metta.api import PPOConfig

ppo_config = PPOConfig(
    clip_coef=0.1,        # Policy clipping coefficient
    ent_coef=0.01,        # Entropy bonus coefficient
    gamma=0.99,           # Discount factor
    gae_lambda=0.95,      # GAE lambda
    vf_coef=0.5,          # Value function loss coefficient
    max_grad_norm=0.5,    # Gradient clipping
)
```

### OptimizerConfig
Optimizer settings:

```python
from metta.api import OptimizerConfig

optimizer_config = OptimizerConfig(
    type="adam",          # "adam" or "muon"
    learning_rate=3e-4,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    weight_decay=0.0,
)
```

### SimulationConfig
Simulation and evaluation settings:

```python
from metta.api import SimulationConfig

simulation_config = SimulationConfig(
    evaluate_interval=300,    # How often to evaluate (epochs)
    replay_interval=300,      # How often to generate replays
    replay_dir="./replays",   # Where to save replays
)
```

### Note on Defaults

All config classes come with carefully chosen defaults based on extensive empirical testing. The defaults are documented in the source code with explanations of why each value was chosen. For example:

- `PPOConfig.clip_coef=0.1`: Conservative value from PPO paper for stability
- `OptimizerConfig.learning_rate=0.0004573...`: Specific value found through hyperparameter sweeps
- `VTraceConfig.vtrace_rho_clip=1.0`: Standard for on-policy training from IMPALA paper

You can confidently use the defaults for most experiments, only overriding values when you have specific requirements.

## Core Components

### Environment
Factory for creating vectorized MettaGrid environments with simplified interface:

```python
from metta.api import Environment

# Full control
env = Environment(
    curriculum_path="/env/mettagrid/simple",
    num_envs=64,
    num_workers=4,
    batch_size=16,
    device="cuda",
    zero_copy=True,
)

# Convenience parameters
env = Environment(
    num_agents=4,    # Quick setup without curriculum
    width=32,
    height=32,
    device="cuda",
    num_envs=64,
)
```

### Agent
Factory for creating Metta agents:

```python
from omegaconf import DictConfig
from metta.api import Agent

# Create agent with full component configuration
agent_config = DictConfig({
    "device": "cuda",
    "agent": {
        "clip_range": 0,
        "analyze_weights_interval": 300,
        "observations": {"obs_key": "grid_obs"},
        "components": {
            # ... component definitions ...
        }
    }
})

agent = Agent(env, agent_config, device="cuda")
```

### TrainingComponents
Container exposing all internal training components:

```python
from metta.api import TrainingComponents

training = TrainingComponents.create(
    vecenv=env,
    policy=agent,
    trainer_config=trainer_config,
    policy_store=policy_store,
)

# Access to all components:
# training.experience - Experience buffer
# training.optimizer - Torch optimizer
# training.losses - Loss tracking
# training.kickstarter - Kickstarter for teacher distillation
# training.timer - Performance timing
```

## Training Loop Control

The API exposes the actual training loop components, giving you full control:

### Rollout Phase
```python
# Reset for new rollout
training.reset_for_rollout()

# Collect experience
raw_infos = []
while not training.is_ready_for_training():
    num_steps, info = training.rollout_step()
    training.agent_step += num_steps
    if info:
        raw_infos.extend(info)

# Process statistics
training.accumulate_stats(raw_infos)
```

### Training Phase
```python
# Reset training state
training.reset_training_state()

# Compute advantages
advantages = training.compute_advantages()

# Calculate annealed beta for prioritized replay
anneal_beta = calculate_anneal_beta(
    epoch=training.epoch,
    total_timesteps=trainer_config.total_timesteps,
    batch_size=trainer_config.batch_size,
    prio_alpha=prio_cfg.prio_alpha,
    prio_beta0=prio_cfg.prio_beta0,
)

# Train on minibatches
for epoch in range(trainer_config.update_epochs):
    for i in range(training.experience.num_minibatches):
        # Sample minibatch with prioritized replay
        minibatch = training.sample_minibatch(
            advantages, i, total_minibatches, anneal_beta
        )

        # Compute loss and update
        loss = training.train_minibatch(minibatch, advantages)
        training.optimize_step(loss, accumulate_steps)
```

## Advanced Features

### Policy Evaluation

Evaluate policies on multiple environments:

```python
from metta.sim.simulation_suite import SimulationSuite
from metta.sim.simulation_config import SimulationSuiteConfig, SingleEnvSimulationConfig

# Define evaluation environments
evaluation_config = SimulationSuiteConfig(
    name="evaluation",
    simulations={
        "navigation/simple": SingleEnvSimulationConfig(
            env="/env/mettagrid/simple",
            num_episodes=5,
            max_time_s=30,
        ),
        "navigation/medium": SingleEnvSimulationConfig(
            env="/env/mettagrid/medium",
            num_episodes=5,
            max_time_s=30,
        ),
    },
)

# Run evaluation
sim_suite = SimulationSuite(
    config=evaluation_config,
    policy_pr=policy_record,
    policy_store=policy_store,
    device=device,
    vectorization="serial",
)
results = sim_suite.simulate()
```

### Replay Generation

Generate replays for visualization:

```python
from metta.sim.simulation import Simulation

replay_config = SingleEnvSimulationConfig(
    env="/env/mettagrid/simple",
    num_episodes=1,
    max_time_s=60,
)

replay_sim = Simulation(
    name=f"replay_epoch_{epoch}",
    config=replay_config,
    policy_pr=policy_record,
    policy_store=policy_store,
    device=device,
    replay_dir="./replays",
)

results = replay_sim.simulate()
replay_urls = results.stats_db.get_replay_urls()
```

### System Monitoring

Monitor system resources during training:

```python
from metta.common.profiling.memory_monitor import MemoryMonitor
from metta.common.util.system_monitor import SystemMonitor

# Memory monitoring
memory_monitor = MemoryMonitor()
memory_monitor.add(training, name="TrainingComponents", track_attributes=True)
memory_monitor.add(agent, name="Agent")

# System monitoring
system_monitor = SystemMonitor(
    sampling_interval_sec=1.0,
    history_size=100,
    auto_start=True,
)

# Get stats
system_stats = system_monitor.get_summary()
memory_stats = memory_monitor.stats()
```

### Gradient Statistics

Track gradient behavior:

```python
from metta.rl.functions import compute_gradient_stats

grad_stats = compute_gradient_stats(agent)
logger.info(
    f"Gradient stats - mean: {grad_stats['grad/mean']:.2e}, "
    f"variance: {grad_stats['grad/variance']:.2e}, "
    f"norm: {grad_stats['grad/norm']:.2e}"
)
```

## Checkpointing

Save and load policies using PolicyStore:

```python
from metta.agent.policy_store import PolicyStore
from metta.api import save_checkpoint, load_checkpoint

# Create policy store
policy_store_cfg = DictConfig({
    "device": str(device),
    "policy_cache_size": 10,
    "trainer": {
        "checkpoint": {
            "checkpoint_dir": "./checkpoints",
        }
    },
})
policy_store = PolicyStore(cfg=policy_store_cfg, wandb_run=None)

# Save checkpoint
policy_record = save_checkpoint(
    policy=agent,
    policy_store=policy_store,
    epoch=100,
    metadata={"score": 0.95}
)

# Load checkpoint
loaded_record = load_checkpoint(policy_store, "path/to/checkpoint")
```

## Object Type Constants

The API exports constants for MettaGrid object types:

```python
from metta.api import (
    TYPE_AGENT,
    TYPE_WALL,
    TYPE_MINE_RED,
    TYPE_GENERATOR_RED,
    TYPE_ALTAR,
    # ... etc
)
```

## Complete Training Example

See `run.py` for a complete example that demonstrates:
- Creating environments and agents
- Setting up training components
- Running the full training loop
- Policy evaluation on multiple environments
- Replay generation with MetaScope URLs
- Memory and system monitoring
- Gradient statistics computation
- Checkpointing and restoration

## Comparison to Full Trainer

| Feature | API (run.py) | Full Trainer (train.py) |
|---------|--------|------------------------|
| Core Training | ✅ | ✅ |
| Policy Evaluation | ✅ | ✅ |
| Replay Generation | ✅ | ✅ |
| Memory Monitoring | ✅ | ✅ |
| System Monitoring | ✅ | ✅ |
| Gradient Stats | ✅ | ✅ |
| Wandb Integration | ❌ | ✅ |
| Distributed Training | ❌ | ✅ |
| Stats Client | ❌ | ✅ |
| Torch Profiler | ❌ | ✅ |
| Hydra Config | ❌ | ✅ |

## Key Differences from Hydra-based Training

1. **Direct instantiation** - No Hydra configuration files needed
2. **Pydantic configs** - Type-safe configuration with validation
3. **Exposed components** - Direct access to Experience, Losses, etc.
4. **Manual control** - You control when rollouts happen, when to train, when to save
5. **Same infrastructure** - Uses the exact same components as MettaTrainer internally

## Implementation Notes

The API is designed to be a thin wrapper around Metta's core functionality:
- `Environment` wraps `make_vecenv` and curriculum creation
- `Agent` wraps `make_policy` with proper initialization
- `TrainingComponents` exposes the internal components that `MettaTrainer` uses
- All training functions (`perform_rollout_step`, `process_minibatch_update`, etc.) are the actual implementations from `metta.rl.functions`

This ensures that training behavior is identical whether you use the API or the traditional Hydra-based approach.

## Troubleshooting

### ImportError: No module named 'metta.mettagrid.mettagrid_c'

If you get this error, the package needs to be installed:
```bash
uv sync --inexact
```

Or if you don't have uv:
```bash
pip install -e .
```

### "Unable to cast Python instance to C++ type" Error

This error typically means the C++ extension needs to be rebuilt. This can happen after:
- Pulling changes from git
- Switching branches
- Updating dependencies

To fix, reinstall the package:
```bash
uv sync --inexact --reinstall-package metta-mettagrid
```

### Module Import Errors After Merge

If imports fail after merging from main, module paths may have changed. Common relocations:
- `metta.common.stopwatch` → `metta.common.profiling.stopwatch`
- `metta.common.memory_monitor` → `metta.common.profiling.memory_monitor`

## Benefits

1. **Transparency**: See exactly how training works without Hydra magic
2. **Hackability**: Easily modify any part of the training loop
3. **Control**: Full control over the training process
4. **Learning**: Great for understanding Metta's internals
5. **Integration**: Easy to integrate with existing codebases

This makes it an excellent starting point for researchers and engineers who want full control over the Metta training process while still having access to advanced features like evaluation and monitoring.
