# Movement Metrics

This document describes the movement metrics feature that tracks agent navigation behavior.

## Overview

Movement metrics track agent behavior related to movement and orientation:
1. **Direction Distribution**: How much time agents spend facing each direction (up, down, left, right)
2. **Sequential Rotation Behavior**: How often agents rotate multiple times in sequence (indicating search/indecision)

## Metrics Collected

### Direction Metrics
- `movement/facing/up` - Steps spent facing up
- `movement/facing/down` - Steps spent facing down
- `movement/facing/left` - Steps spent facing left
- `movement/facing/right` - Steps spent facing right

### Sequential Behavior Metrics
- `movement/sequential_rotations` - Average length of sequential rotation sequences (indicates search/indecision patterns)

**Note**: Action usage metrics (`action.rotate.success`, `action.move.success`, etc.) are already tracked by the existing action system and provide more detailed information than movement metrics would.

## Enabling Movement Metrics

### 1. Configuration File
Add to your game configuration:
```yaml
game:
  track_movement_metrics: true
```

### 2. Training Scripts
For tools/train.py:
```bash
./tools/train.py run=$USER.test_train +hardware=macbook +trainer.env=env/mettagrid/simple +trainer.env_overrides.game.track_movement_metrics=true
```

### 3. Navigation Recipe
Use the MOVEMENT_METRICS environment variable:
```bash
MOVEMENT_METRICS=true ./recipes/navigation.sh
```

### 4. API Usage
```python
from metta.api import Environment

# Enable movement metrics
env = Environment(
    num_agents=4,
    track_movement_metrics=True
)
```

## Performance Impact

Movement metrics add minimal computational overhead:
- **Computational cost**: O(num_agents) per step
- **Memory cost**: Negligible (just counter storage)
- **Expected overhead**: < 5% in most cases

Run the performance test to measure impact on your setup:
```bash
python test_movement_metrics_performance.py
```

## WandB Integration

Movement metrics automatically appear in WandB logs under:
- `env_agent/movement/facing/up` - Steps spent facing up
- `env_agent/movement/facing/down` - Steps spent facing down
- `env_agent/movement/facing/left` - Steps spent facing left
- `env_agent/movement/facing/right` - Steps spent facing right
- `env_agent/movement/sequential_rotations` - Sequential rotation behavior

Action usage is already tracked under:
- `env_agent/action.rotate.success/failed` - Rotation action outcomes
- `env_agent/action.move.success/failed` - Movement action outcomes
- `env_agent/action.*.success/failed` - All other action outcomes

The metrics include all standard aggregations (mean, std_dev, rate, etc.) provided by the stats system.

**Note**: These metrics are **disabled by default** for backwards compatibility and performance. They must be explicitly enabled using one of the methods above.

## Use Cases

### Research Applications
- **Exploration Analysis**: Track how agents explore environments
- **Behavior Characterization**: Identify systematic vs. random movement patterns
- **Training Dynamics**: Monitor how movement patterns change during training
- **Decision Making**: Use existing action metrics to analyze action preferences

### Debugging
- **Indecisive Agents**: High `sequential_rotations` values indicate agents that can't decide on direction
- **Biased Movement**: Uneven direction distributions may reveal environment biases
- **Inefficient Exploration**: High sequential rotation rates may indicate poor exploration strategies
- **Action Balance**: Use existing `action.*.success` metrics for detailed action analysis

## Testing

Test the metrics with:
```bash
# Basic functionality test
python test_navigation_metrics.py

# Performance comparison
python test_movement_metrics_performance.py

# API integration test
python test_movement_metrics_api.py
```

## Summary

Movement metrics provide focused insights into agent navigation behavior:
- **Direction distribution**: How agents orient themselves in the environment
- **Sequential rotation behavior**: Patterns that indicate search or indecision
- **Minimal overhead**: Only active when explicitly enabled
- **WandB integration**: Automatically logged with full statistical analysis

These metrics complement existing action metrics to provide a complete picture of agent navigation behavior.

## Backwards Compatibility

Movement metrics are disabled by default (`track_movement_metrics: false`), ensuring full backwards compatibility with existing configurations and scripts.
