# Performance Threshold Tracking

This feature tracks when smoothed performance metrics reach specified thresholds and calculates the associated samples, time, and cost to reach them.

## Overview

Performance threshold tracking monitors training progress and determines when agents achieve specific performance targets. For the arena environment, it tracks when `env_agent/heart.gained` reaches 2.0 and 5.0 hearts.

## Features

- **Smoothing**: Uses exponential moving average to smooth noisy metrics
- **Multiple Thresholds**: Track multiple performance targets simultaneously
- **Cost Calculation**: Estimates AWS costs based on instance type and time
- **WandB Integration**: Logs results to WandB for visualization
- **Extensible**: Easy to add new thresholds via configuration files

## Arena Environment Thresholds

The arena environment tracks two performance thresholds:

1. **heart_gained_2**: `env_agent/heart.gained >= 2.0`
2. **heart_gained_5**: `env_agent/heart.gained >= 5.0`

## Metrics Tracked

For each threshold, the following metrics are logged to WandB:

- `performance_threshold/{threshold_name}/metric`: The metric being tracked
- `performance_threshold/{threshold_name}/target_value`: The target value
- `performance_threshold/{threshold_name}/achieved`: Whether threshold was reached
- `performance_threshold/{threshold_name}/final_smoothed_value`: Final smoothed value
- `performance_threshold/{threshold_name}/samples_to_threshold`: Samples needed (NaN if not reached)
- `performance_threshold/{threshold_name}/minutes_to_threshold`: Time needed (NaN if not reached)
- `performance_threshold/{threshold_name}/cost_to_threshold`: Estimated AWS cost (NaN if not reached)

## Usage

### Training with Performance Tracking

The performance threshold tracking is automatically enabled for arena training. Use the modified recipe:

```bash
./recipes/arena_with_performance_tracking.sh
```

### Configuration

Performance thresholds are configured in `configs/performance_thresholds/arena.yaml`:

```yaml
thresholds:
  - name: "heart_gained_2"
    metric: "env_agent/heart.gained"
    target_value: 2.0
    smoothing_factor: 0.1

  - name: "heart_gained_5"
    metric: "env_agent/heart.gained"
    target_value: 5.0
    smoothing_factor: 0.1

aws:
  instance_type: "g5.4xlarge"
  use_spot: false
```

### Adding New Thresholds

To add new performance thresholds:

1. **For new environments**: Create a new configuration file in `configs/performance_thresholds/`
2. **For existing environments**: Modify the existing configuration file
3. **Update the trainer**: Modify `_init_performance_threshold_tracker()` in `metta/rl/trainer.py`

Example for adding a new threshold:

```yaml
thresholds:
  - name: "heart_gained_10"
    metric: "env_agent/heart.gained"
    target_value: 10.0
    smoothing_factor: 0.1
```

## AWS Cost Calculation

The system estimates costs based on AWS pricing:

- **Instance Types**: Supports g4dn.xlarge, g5.xlarge, g5.2xlarge, g5.4xlarge, g5.8xlarge, g5.12xlarge, g5.24xlarge, p3.2xlarge, p3.8xlarge, p3.16xlarge
- **Pricing**: Uses on-demand pricing by default, supports spot instances
- **Calculation**: `cost = hours * price_per_hour`

## Visualization

In WandB, you can create visualizations showing:

1. **Threshold Achievement Timeline**: When each threshold was reached
2. **Cost Efficiency**: Cost vs. performance comparison
3. **Training Efficiency**: Samples/time needed to reach thresholds

## Example WandB Queries

```python
# Get all performance threshold metrics
wandb.log({"performance_threshold/heart_gained_2/samples_to_threshold": samples})

# Check if threshold was achieved
if wandb.run.summary.get("performance_threshold/heart_gained_2/achieved"):
    print("Heart gained 2 threshold achieved!")
```

## Implementation Details

### Core Components

- `PerformanceThresholdTracker`: Main tracking class
- `PerformanceThreshold`: Configuration dataclass
- `ThresholdResult`: Result dataclass
- Configuration loading utilities

### Integration Points

- **Trainer**: Integrated into `MettaTrainer._process_stats()`
- **WandB**: Metrics logged via `get_wandb_metrics()`
- **Configuration**: YAML-based configuration system

### Smoothing Algorithm

Uses exponential moving average:
```
smoothed_value = α * current_value + (1 - α) * previous_smoothed_value
```

Where α is the smoothing factor (default: 0.1).

## Testing

Run the test script to verify functionality:

```bash
python test_arena_performance_tracking.py
```

This simulates arena training and verifies threshold tracking works correctly.
