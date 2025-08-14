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

### Using the Environmental Toggle (Recommended)

The easiest way to enable task-scaled performance is using the `enable_task_perf_target` toggle:

```yaml
# In your trainer config or environment overrides
trainer:
  env_overrides:
    enable_task_perf_target: true
```

When this toggle is enabled:
- **Automatic reward targets**: If no `reward_target` is explicitly set, the system automatically generates one based on the task ID
- **Deterministic generation**: The same task ID always produces the same reward target
- **Configurable ranges**: Default range is 0.0 to 10.0, but can be customized

### Configurable Reward Target Ranges

You can customize the range for auto-generated reward targets:

```yaml
# In your trainer config or environment overrides
trainer:
  env_overrides:
    enable_task_perf_target: true
    reward_target_min: 1.0    # Minimum value for auto-generated targets
    reward_target_max: 25.0   # Maximum value for auto-generated targets
```

Or in your curriculum config:

```yaml
# In your curriculum config
_target_: metta.mettagrid.curriculum.bucketed.BucketedCurriculum

env_cfg_template_path: /env/mettagrid/navigation/training/terrain_from_numpy_defaults

# Enable task-scaled performance with custom range
enable_task_perf_target: true
reward_target_min: 5.0
reward_target_max: 20.0

buckets:
  game.map_builder.instance_map.params.dir:
    - terrain_maps_nohearts
    - varied_terrain/balanced_large
```

**Default values:**
- `reward_target_min`: 0.0
- `reward_target_max`: 10.0

### Manual Configuration

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

### Example Usage

#### Simple Toggle Usage
```bash
# Enable task-scaled performance for any curriculum
./devops/skypilot/launch.py train \
  trainer.curriculum=/env/mettagrid/curriculum/navigation/bucketed \
  trainer.env_overrides.enable_task_perf_target=true \
  wandb.project=metta \
  wandb.group=task_scaled_perf_test
```

#### With Custom Reward Targets
```yaml
# In your curriculum config
_target_: metta.mettagrid.curriculum.bucketed.BucketedCurriculum

env_cfg_template_path: /env/mettagrid/navigation/training/terrain_from_numpy_defaults

buckets:
  game.map_builder.instance_map.params.dir:
    - terrain_maps_nohearts
    - varied_terrain/balanced_large

# Enable the toggle for automatic reward targets
enable_task_perf_target: true

# Or specify custom reward targets
reward_target_bucket: [1.0, 1.5, 2.0, 2.5, 3.0]
```

## Usage in Curriculum Systems

### Accessing Task-Scaled Performance

The `task_scaled_performance` metric is automatically calculated and available in the environment's `infos` dictionary:

```python
# In your training loop or evaluation code
observations, rewards, terminals, truncations, infos = env.step(actions)

if "task_scaled_performance" in infos:
    for task_id, performance in infos["task_scaled_performance"].items():
        print(f"Task {task_id}: {performance:.2f} performance")
```

### Integration with Learning Progress Curricula

Task-scaled performance works seamlessly with learning progress curricula:

```yaml
# Learning progress curriculum with task-scaled performance
_target_: metta.mettagrid.curriculum.learning_progress.LearningProgressCurriculum

tasks:
  /env/mettagrid/navigation/training/terrain_from_numpy: 1
  /env/mettagrid/navigation/training/cylinder_world: 1

env_overrides:
  enable_task_perf_target: true
```

### Integration with Prioritized Regression Curricula

```yaml
# Prioritized regression curriculum with task-scaled performance
_target_: metta.mettagrid.curriculum.prioritize_regressed.PrioritizeRegressedCurriculum

tasks:
  /env/mettagrid/navigation/training/terrain_from_numpy: 1
  /env/mettagrid/navigation/training/cylinder_world: 1

env_overrides:
  enable_task_perf_target: true
```

## Implementation Details

### Automatic Reward Target Generation

When `enable_task_perf_target` is enabled but no `reward_target` is set:

```python
# Deterministic generation based on task_id and configurable range
min_val = env_config.reward_target_min
max_val = env_config.reward_target_max
rng.seed(task_id)  # Ensure deterministic sampling
reward_target = rng.uniform(min_val, max_val)
```

This ensures:
- **Determinism**: Same task_id always produces same reward_target
- **Variety**: Different tasks have different difficulty levels
- **Configurable range**: Targets can be customized via `reward_target_min` and `reward_target_max`
- **Uniform distribution**: Targets are sampled uniformly from the specified range

### Performance Calculation

The calculation happens in `CurriculumEnv.step()`:

```python
if env_cfg.enable_task_perf_target and env_cfg.reward_target is not None:
    if env_cfg.reward_target > 0:
        task_scaled_performance = min(mean_reward / env_cfg.reward_target, 1.0)
```

### Logging

Task-scaled performance is automatically logged to WandB when available:

- **Metric name**: `task_scaled_performance`
- **Format**: Dictionary mapping task_id to performance value
- **Range**: 0.0 to 1.0 (capped at 100% performance)

## Migration Guide

### From Manual Configuration

If you were previously using manual `reward_target` configuration:

**Before:**
```yaml
reward_target_bucket: [5.0, 10.0, 15.0, 20.0, 25.0]
```

**After:**
```yaml
enable_task_perf_target: true
# Or keep your custom targets:
reward_target_bucket: [5.0, 10.0, 15.0, 20.0, 25.0]
```

### Backward Compatibility

The system maintains full backward compatibility:
- Existing configs without the toggle continue to work unchanged
- Manual `reward_target` settings take precedence over auto-generation
- The toggle only affects behavior when explicitly enabled
