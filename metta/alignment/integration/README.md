# Metta Integration for GAMMA Metrics

This module provides integration between GAMMA alignment metrics and Metta's training framework.

## Components

### 1. TrajectoryCollector

Collects agent positions, velocities, and task directions during episodes.

```python
from metta.alignment.integration import TrajectoryCollector

collector = TrajectoryCollector(num_agents=16)

# Reset at episode start
collector.reset()

# Record each step
for step in range(episode_length):
    positions = extract_positions(env)  # shape (16, 2)
    task_dirs = compute_task_directions(env, positions)
    collector.record_step(positions, task_directions=task_dirs, dt=0.1)

# Get trajectories
trajectories = collector.get_trajectories()
```

### 2. MettaGridAdapter

Extracts positions and computes task directions from MettaGrid environments.

```python
from metta.alignment.integration import MettaGridAdapter

adapter = MettaGridAdapter(grid_to_continuous_scale=1.0)

# Extract agent positions from environment
positions = adapter.extract_agent_positions(env)

# Compute task directions (e.g., toward resources)
task_dirs = adapter.compute_task_directions_to_resources(
    env,
    resource_types=['generator', 'converter']
)

# Or toward specific goals
task_dirs = adapter.compute_task_directions_to_goal(positions, goals)
```

### 3. GAMMAEvaluator

Computes GAMMA metrics from collected trajectories.

```python
from metta.alignment.integration import GAMMAEvaluator

evaluator = GAMMAEvaluator(alpha=0.1)

# Evaluate alignment
results = evaluator.evaluate(trajectories, dt=0.1, goals=goal_positions)

print(f"GAMMA: {results['GAMMA']:.3f}")
print(f"Mean IAM: {results['IAM_mean']:.3f}")

# Get detailed component breakdown
detailed = evaluator.evaluate_with_components(trajectories, dt=0.1)
for i, comp in enumerate(detailed['components']):
    print(f"Agent {i}: A={comp['A']:.2f}, D={comp['D']:.2f}, E={comp['E']:.2f}")
```

### 4. GAMMALogger (TrainerComponent)

Integrates GAMMA into Metta's training loop.

```python
from metta.alignment.integration import GAMMALogger

# Add to trainer components
gamma_logger = GAMMALogger(
    num_agents=16,
    epoch_interval=10,  # Compute every 10 epochs
    alpha=0.1,
    enabled=True
)

# The component will automatically:
# - Collect trajectories during rollouts
# - Compute GAMMA at epoch boundaries
# - Log to wandb and console
```

## Integration Example

Here's a complete example of integrating GAMMA into a Metta training run:

```python
from metta.rl.training import Trainer
from metta.alignment.integration import GAMMALogger

# Create trainer with GAMMA logging
trainer = Trainer(config)

# Add GAMMA logger component
gamma_logger = GAMMALogger(
    num_agents=trainer.config.num_agents,
    epoch_interval=10,
    alpha=0.1,
    enabled=True
)
trainer.add_component(gamma_logger)

# Train normally - GAMMA metrics will be logged automatically
trainer.train()

# Get summary statistics
summary = gamma_logger.get_summary_stats()
print(f"Final GAMMA: {summary['alignment/GAMMA_final']:.3f}")
```

## Manual Trajectory Collection

For evaluation or debugging, you can manually collect and evaluate trajectories:

```python
from metta.alignment.integration import (
    TrajectoryCollector,
    MettaGridAdapter,
    GAMMAEvaluator
)

# Setup
collector = TrajectoryCollector(num_agents=16)
adapter = MettaGridAdapter()
evaluator = GAMMAEvaluator()

# Run episode
obs, info = env.reset()
collector.reset()

for step in range(max_steps):
    # Extract positions
    positions = adapter.extract_agent_positions(env)

    # Compute task directions
    task_dirs = adapter.compute_task_directions_to_resources(env)

    # Record
    collector.record_step(positions, task_directions=task_dirs, dt=0.1)

    # Step environment
    actions = policy(obs)
    obs, rewards, dones, truncs, info = env.step(actions)

    if dones.all() or truncs.all():
        break

# Evaluate
trajectories = collector.get_trajectories()
results = evaluator.evaluate(trajectories, dt=0.1)

print(f"Episode GAMMA: {results['GAMMA']:.3f}")
```

## Computing Task Directions

Task directions depend on the task type:

### Resource Collection
```python
# Point toward nearest generator/converter
task_dirs = adapter.compute_task_directions_to_resources(
    env,
    resource_types=['generator', 'converter']
)
```

### Goal Reaching
```python
# Point toward specific goal positions
goals = np.array([[10, 10], [20, 20], ...])  # One per agent
task_dirs = adapter.compute_task_directions_to_goal(positions, goals)
```

### Formation
```python
# Point toward desired formation positions
desired_formation = compute_formation_positions(...)
task_dirs = adapter.compute_task_directions_formation(positions, desired_formation)
```

### Custom Task Field
```python
# Define custom task field
def custom_task_field(position):
    # Return unit direction vector
    return direction / np.linalg.norm(direction)

task_dirs = np.array([custom_task_field(pos) for pos in positions])
```

## Logging to Wandb

GAMMA metrics are automatically formatted for wandb:

```python
results = evaluator.evaluate(trajectories, dt=0.1)
wandb_dict = evaluator.format_for_wandb(results, prefix="alignment")

# Logs:
# - alignment/GAMMA
# - alignment/GAMMA_alpha
# - alignment/IAM_mean
# - alignment/IAM_std
# - alignment/CV
# - alignment/agent_0/IAM
# - alignment/agent_1/IAM
# - ...
# - alignment/mean_A (goal attainment)
# - alignment/mean_D (directional intent)
# - alignment/mean_E (path efficiency)
# - alignment/mean_T (time efficiency)
# - alignment/mean_Y (energy proportionality)
```

## Next Steps

To complete the integration:

1. **Hook trajectory collection into environment step**
   - Modify `MettaGridEnv.step()` to expose positions
   - Or use `grid_objects()` to extract positions each step

2. **Define task-specific direction computation**
   - CvC: toward resources, facilities, teammates
   - Custom tasks: implement task interface

3. **Add to default trainer configuration**
   - Make GAMMA logging opt-in via config flag
   - Set appropriate epoch interval

4. **Create evaluation script**
   - Standalone script to evaluate trained policies
   - Generate alignment reports

## Performance Considerations

- GAMMA computation is relatively lightweight (O(T·N) where T=timesteps, N=agents)
- Trajectory storage is O(T·N·d) where d=dimension
- Consider computing every 10-100 epochs rather than every epoch
- Can disable during training and enable only for evaluation
