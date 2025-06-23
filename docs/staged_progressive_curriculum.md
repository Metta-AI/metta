# Staged Progressive Curriculum

The `StagedProgressiveCurriculum` is a new curriculum type that replaces multi-stage training approaches by automatically transitioning between different curriculum stages within a single training run.

## Overview

Instead of running separate training commands for different stages (e.g., navigation → navigation sequence → simple tasks), the staged progressive curriculum automatically manages transitions between stages based on performance or time thresholds.

## Key Benefits

1. **Single Training Run**: No need for multiple bash commands or manual stage management
2. **Automatic Transitions**: Stages transition automatically based on performance or time
3. **Complete History**: The full training recipe is captured in a single agent's history
4. **WandB Logging**: All stage transitions and performance metrics are logged to WandB
5. **Flexible Configuration**: Support for both performance-based and time-based transitions

## Configuration

### Basic Configuration

```yaml
_target_: mettagrid.curriculum.StagedProgressiveCurriculum

stages:
  # Stage 1: Navigation training
  - curriculum: /env/mettagrid/curriculum/navigation
    name: "navigation_training"
    weight: 1.0
  
  # Stage 2: Navigation sequence training  
  - curriculum: /env/mettagrid/curriculum/navsequence/nav_backchain
    name: "navigation_sequence"
    weight: 1.0
  
  # Stage 3: Simple tasks
  - curriculum: /env/mettagrid/curriculum/simple
    name: "simple_tasks"
    weight: 1.0

# Transition configuration
transition_criteria: "performance"  # "performance" or "time"
performance_threshold: 0.8  # Transition when performance reaches 80%
time_threshold_steps: 100000  # Alternative: transition after 100k steps
transition_smoothing: 0.1  # Smoothing factor for performance tracking
```

### Stage Configuration Options

Each stage can be configured as either:

1. **Simple string**: Just the curriculum path
   ```yaml
   stages:
     - "/env/mettagrid/curriculum/navigation"
   ```

2. **Detailed dict**: With name and weight
   ```yaml
   stages:
     - curriculum: "/env/mettagrid/curriculum/navigation"
       name: "navigation_training"
       weight: 1.0
   ```

### Transition Criteria

#### Performance-based Transitions
- Transitions occur when the agent's performance in the current stage reaches a threshold
- Performance is tracked using exponential moving average with smoothing
- Default threshold: 0.8 (80% success rate)

#### Time-based Transitions
- Transitions occur after a specified number of training steps
- Useful for ensuring minimum training time per stage
- Default: 100,000 steps per stage

## WandB Logging

The curriculum automatically logs the following metrics to WandB:

- `curriculum/current_stage`: Current stage index
- `curriculum/stage_name`: Current stage name
- `curriculum/stage_probs`: Probability distribution over stages (one-hot for current stage)
- `curriculum/stage_performance`: Performance metrics for each stage
- `curriculum/stage_transition`: Detailed transition events with metadata

## Usage Example

### Training Command

```bash
# Single command for multi-stage training
./devops/skypilot/launch.py train run=USER.staged_progressive_example \
  trainer.curriculum=env/mettagrid/curriculum/staged_progressive_example \
  --no-spot --gpus=4 --skip-git-check
```

### User Configuration

```yaml
# configs/user/staged_progressive_example.yaml
defaults:
  - /trainer/trainer
  - /sim/all@evals
  - _self_

run: ${oc.env:USER}.staged_progressive_example
trainer:
  curriculum: env/mettagrid/curriculum/staged_progressive_example
  total_timesteps: 2000000  # Longer training for stage transitions

wandb:
  enabled: true
  project: metta
  entity: metta-research
  tags: ["staged_progressive", "multi_stage_training"]
```

## Migration from Multi-stage Training

### Before (Multi-stage)
```bash
# Stage 1: Navigation
./devops/skypilot/launch.py train run=USER.navigation \
  trainer.curriculum=env/mettagrid/curriculum/navigation

# Stage 2: Navigation Sequence  
./devops/skypilot/launch.py train run=USER.sequence \
  trainer.curriculum=env/mettagrid/curriculum/navsequence/nav_backchain \
  trainer.initial_policy.uri=wandb://run/USER.navigation

# Stage 3: Simple Tasks
./devops/skypilot/launch.py train run=USER.simple \
  trainer.curriculum=env/mettagrid/curriculum/simple \
  trainer.initial_policy.uri=wandb://run/USER.sequence
```

### After (Staged Progressive)
```bash
# Single command for all stages
./devops/skypilot/launch.py train run=USER.staged_progressive \
  trainer.curriculum=env/mettagrid/curriculum/staged_progressive_example
```

## Advanced Configuration

### Custom Transition Logic

You can customize transition behavior by modifying the curriculum class:

```python
class CustomStagedCurriculum(StagedProgressiveCurriculum):
    def _should_transition_to_next_stage(self) -> bool:
        # Custom transition logic
        current_performance = self._stage_performance[self._current_stage]
        current_stage_steps = self._total_steps - self._stage_start_step
        
        # Transition if both performance and time criteria are met
        performance_ready = current_performance >= self.performance_threshold
        time_ready = current_stage_steps >= self.time_threshold_steps
        
        return performance_ready and time_ready
```

### Environment Overrides

Environment overrides are applied to all stages:

```yaml
env_overrides:
  game:
    num_agents: 24
    max_steps: 1000
  sampling: 0.7
```

## Monitoring and Debugging

### WandB Dashboard

Monitor training progress in the WandB dashboard:

1. **Stage Transitions**: Look for `curriculum/stage_transition` events
2. **Performance Tracking**: Monitor `curriculum/stage_performance` for each stage
3. **Current Stage**: Track `curriculum/current_stage` over time

### Logs

The curriculum logs transition events:

```
INFO - Transitioning from stage 0 (navigation_training) to stage 1 (navigation_sequence) at step 150000
```

## Testing

Run the curriculum tests:

```bash
pytest mettagrid/tests/test_staged_progressive_curriculum.py -v
```

## Limitations

1. **Sequential Stages**: Stages progress in order (no branching or parallel stages)
2. **Performance Dependency**: Performance-based transitions require reliable performance metrics
3. **Single Agent**: All stages must use the same number of agents

## Future Enhancements

Potential improvements for future versions:

1. **Parallel Stages**: Support for multiple active stages
2. **Branching Logic**: Conditional stage transitions based on multiple criteria
3. **Adaptive Thresholds**: Dynamic adjustment of transition thresholds
4. **Stage-specific Configurations**: Different hyperparameters per stage 