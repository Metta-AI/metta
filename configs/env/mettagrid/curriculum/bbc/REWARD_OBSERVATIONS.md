# BBC Curriculum with Reward Observations

The BBC (Learning Progress Curriculum) has been updated to support detailed reward observations instead of just using a single aggregated score. This allows the curriculum to track learning progress for each reward type separately, providing more granular control over task selection.

## Features

- **Multi-Reward Tracking**: Track learning progress separately for each reward type (ore_red, battery_red, heart, laser, armor, blueprint)
- **Flexible Aggregation**: Choose how to combine learning progress across reward types (mean, max, or sum)
- **Backward Compatible**: Still supports single score values for compatibility with existing code

## Configuration

To enable reward observations in the BBC curriculum, the following parameters have been added to `bbc.yaml`:

```yaml
# Enable reward observation tracking
use_reward_observations: true
reward_types: ["ore_red", "battery_red", "heart", "laser", "armor", "blueprint"]
reward_aggregation: "mean"  # How to aggregate learning progress across reward types
```

### Parameters

- `use_reward_observations`: Set to `true` to enable detailed reward tracking
- `reward_types`: List of reward types to track (should match the rewards defined in your tasks)
- `reward_aggregation`: How to combine learning progress scores:
  - `"mean"`: Average the learning progress across all reward types
  - `"max"`: Use the maximum learning progress from any reward type
  - `"sum"`: Sum the learning progress across all reward types

## How It Works

1. **Episode Completion**: When an episode completes, the environment extracts detailed reward statistics from the agent stats (e.g., "ore_red.gained", "battery_red.gained")

2. **Reward Observations**: These statistics are normalized by the number of agents and passed to the curriculum as a dictionary:
   ```python
   reward_observations = {
       "ore_red": 0.5,
       "battery_red": 0.8,
       "heart": 0.2,
       "total": 0.6  # Overall episode reward
   }
   ```

3. **Learning Progress Tracking**: Each reward type has its own learning progress tracker that maintains fast and slow moving averages to detect improvement

4. **Task Selection**: The curriculum aggregates learning progress across all reward types according to the `reward_aggregation` setting, then selects tasks with the highest aggregated learning progress

## Benefits

- **Better Task Selection**: Tasks that show improvement in specific reward types will be prioritized
- **Specialized Learning**: The curriculum can focus on tasks where agents are making progress on particular objectives
- **More Informative**: Provides detailed statistics about learning progress per reward type

## Example Use Cases

1. **Resource Collection Focus**: Set `reward_aggregation: "max"` to prioritize tasks where agents are improving at collecting any resource

2. **Balanced Progress**: Use `reward_aggregation: "mean"` to ensure agents make progress across all reward types

3. **Combat Specialization**: Track only combat-related rewards (laser, armor, heart) to focus on combat skill development

## Monitoring

The curriculum now provides detailed statistics for each reward type:

```
lp/task_success_rate/ore_red: 0.75
lp/task_success_rate/battery_red: 0.82
lp/task_success_rate/heart: 0.45
...
```

These can be monitored in wandb or other logging systems to understand which aspects of the tasks agents are mastering.