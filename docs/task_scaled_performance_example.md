# Task-Scaled Performance Example

This example shows how to use the `enable_task_perf_target` toggle to enable task-scaled performance analysis with minimal configuration changes.

## Quick Start

### Enable for Any Curriculum

To enable task-scaled performance for any existing curriculum, simply add the toggle to your trainer configuration:

```bash
./devops/skypilot/launch.py train \
  trainer.curriculum=/env/mettagrid/curriculum/navigation/bucketed \
  trainer.env_overrides.enable_task_perf_target=true \
  wandb.project=metta \
  wandb.group=task_scaled_perf_test
```

### What Happens When Enabled

When `enable_task_perf_target=true`:

1. **Automatic Reward Targets**: If no `reward_target` is explicitly set, the system automatically generates one based on the task ID
2. **Deterministic Generation**: The same task ID always produces the same reward target
3. **Configurable Range**: Default range is 0.0 to 10.0, but can be customized
4. **Performance Calculation**: `task_scaled_performance = min(reward / reward_target, 1.0)` is calculated and logged
5. **WandB Logging**: The metric appears in your WandB logs as `task_scaled_performance`

### Customizing the Range

You can customize the range for auto-generated reward targets:

```bash
# Enable with custom range
./devops/skypilot/launch.py train \
  trainer.curriculum=/env/mettagrid/curriculum/navigation/bucketed \
  trainer.env_overrides.enable_task_perf_target=true \
  trainer.env_overrides.reward_target_min=1.0 \
  trainer.env_overrides.reward_target_max=25.0 \
  wandb.project=metta \
  wandb.group=task_scaled_perf_test
```

### Example Output

In your WandB logs, you'll see:
```
task_scaled_performance: {
  "42": 0.75,  # Task 42 achieved 75% of target
  "43": 0.50,  # Task 43 achieved 50% of target
  "44": 1.00   # Task 44 achieved 100% of target (capped)
}
```

## Comparison Example

### With Task-Scaled Performance (Toggle On)
```bash
./devops/skypilot/launch.py train \
  trainer.curriculum=/env/mettagrid/curriculum/navigation/bucketed \
  trainer.env_overrides.enable_task_perf_target=true \
  run=nav_with_task_scaled_perf \
  wandb.project=metta \
  wandb.group=task_scaled_perf_comparison
```

### Without Task-Scaled Performance (Toggle Off)
```bash
./devops/skypilot/launch.py train \
  trainer.curriculum=/env/mettagrid/curriculum/navigation/bucketed \
  trainer.env_overrides.enable_task_perf_target=false \
  run=nav_without_task_scaled_perf \
  wandb.project=metta \
  wandb.group=task_scaled_perf_comparison
```

## Custom Reward Targets

If you want to specify custom reward targets instead of using auto-generation:

```yaml
# In your curriculum config
_target_: metta.mettagrid.curriculum.bucketed.BucketedCurriculum

env_cfg_template_path: /env/mettagrid/navigation/training/terrain_from_numpy_defaults

buckets:
  game.map_builder.instance_map.params.dir:
    - terrain_maps_nohearts
    - varied_terrain/balanced_large

# Enable the toggle with custom range
enable_task_perf_target: true
reward_target_min: 1.0
reward_target_max: 5.0

# Or override auto-generation with custom targets
reward_target_bucket: [1.0, 1.5, 2.0, 2.5, 3.0]
```

## Benefits

1. **Zero Config Changes**: Works with any existing curriculum
2. **Automatic Setup**: No need to manually specify reward targets
3. **Deterministic**: Same task always has same target
4. **Normalized Metrics**: All tasks measured on 0-1 scale
5. **Backward Compatible**: Existing configs work unchanged

## Recipes

Use the provided recipes to test the feature:

```bash
# Test with task-scaled performance
./recipes/navigation_task_scaled_performance.sh

# Test without task-scaled performance (for comparison)
./recipes/navigation_original_metric.sh
```
