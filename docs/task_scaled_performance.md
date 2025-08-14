# Task-Scaled Performance

## Overview

Task-scaled performance is a new metric that provides a normalized performance measure from 0 to 1, making it easier to compare performance across tasks with different reward scales and to design curricula that adapt to performance levels.

## How It Works

Task-scaled performance is calculated as:
```
task_scaled_performance = min(reward / reward_target, 1.0)
```

Where:
- `reward` is the actual reward achieved by the agent
- `reward_target` is the target reward for the task (set during task generation)
- The result is capped at 1.0 (100% performance)

## Benefits

1. **Normalized Metrics**: All tasks have performance measured on the same 0-1 scale
2. **Curriculum Design**: Easier to design curricula that sample more challenging targets
3. **Progress Tracking**: Clear indication of how close agents are to optimal performance
4. **Cross-Task Comparison**: Compare performance across tasks with different reward scales

## Configuration

### Setting Reward Targets

#### Fixed Reward Target
```yaml
# In task generator config
reward_target: 10.0
```

#### Sampled Reward Targets
```yaml
# Sample from a range of difficulty levels
reward_target_bucket: [5.0, 10.0, 15.0, 20.0, 25.0]
```

#### Using ValueRange for Continuous Sampling
```yaml
# Sample from a continuous range
reward_target_bucket:
  - range_min: 5.0
    range_max: 25.0
```

### Example Curriculum Configuration

```yaml
_target_: metta.mettagrid.curriculum.bucketed.BucketedCurriculum

env_cfg_template_path: /env/mettagrid/arena/advanced_easy

# Sample reward targets from different difficulty levels
reward_target_bucket: [5.0, 10.0, 15.0, 20.0, 25.0]

# Sample other task parameters
buckets:
  game.agent.rewards.inventory.ore_red: [0, 0.5, 1]
  game.agent.rewards.inventory.battery_red: [0, 0.5, 1]
  game.agent.rewards.inventory.heart: [0, 0.5, 1]

env_overrides:
  game:
    num_agents: 24
    max_steps: 1000
```

## Usage in Curriculum Systems

### Accessing Task-Scaled Performance

The task-scaled performance is automatically calculated and made available in the environment's info dictionary:

```python
# In environment step
obs, rewards, terminals, truncations, infos = env.step(action)

# Task-scaled performance is available in infos
if 'task_scaled_performance' in infos:
    scaled_perf = infos['task_scaled_performance'][task_id]
    print(f"Task {task_id} scaled performance: {scaled_perf}")
```

### Curriculum Adaptation

Curriculum systems can use task-scaled performance to:

1. **Prioritize Underperforming Tasks**: Focus on tasks where scaled performance is low
2. **Progressive Difficulty**: Increase reward targets as performance improves
3. **Performance-Based Sampling**: Sample tasks based on scaled performance rather than raw rewards

### Example: Adaptive Curriculum

```python
class AdaptiveCurriculum(Curriculum):
    def complete_task(self, id: str, score: float, scaled_performance: float):
        # Use scaled performance for task selection
        if scaled_performance < 0.5:
            # Increase weight for tasks with low performance
            self._task_weights[id] *= 1.5
        elif scaled_performance > 0.8:
            # Decrease weight for tasks with high performance
            self._task_weights[id] *= 0.8
```

## Implementation Details

### Environment Configuration

The `EnvConfig` class now includes a `reward_target` field:

```python
class EnvConfig(Config):
    game: GameConfig = Field(default_factory=GameConfig)
    desync_episodes: bool = Field(default=True)
    reward_target: Optional[float] = Field(
        default=None,
        description="Target reward for this task"
    )
```

### Task Generator Integration

Task generators can set reward targets during task generation:

```python
# In BucketedTaskGenerator
if self._config.reward_target_bucket is not None:
    reward_target = self._get_bucket_value(self._config.reward_target_bucket, rng)
    env_config.reward_target = reward_target
```

### Automatic Calculation

The curriculum environment wrapper automatically calculates task-scaled performance:

```python
# In CurriculumEnv.step()
if hasattr(env_cfg, 'reward_target') and env_cfg.reward_target is not None:
    reward_target = env_cfg.reward_target
    if reward_target > 0:
        task_scaled_performance = min(mean_reward / reward_target, 1.0)
```

## Migration Guide

### Existing Configurations

Existing configurations will continue to work without changes. The `reward_target` field is optional and defaults to `None`.

### Adding Task-Scaled Performance

To add task-scaled performance to existing curricula:

1. Add `reward_target` or `reward_target_bucket` to your task generator configuration
2. Update curriculum logic to use scaled performance if desired
3. Monitor the new `task_scaled_performance` metrics in logs

### Backward Compatibility

- All existing reward systems continue to work unchanged
- Task-scaled performance is an additional metric, not a replacement
- Raw rewards are still logged and used by default
