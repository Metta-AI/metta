# Performance Threshold Tracking

This document describes the performance threshold tracking system that monitors training progress and calculates costs when specific performance metrics reach target values.

## Overview

The performance threshold tracking system:

1. **Monitors training metrics** in real-time during training
2. **Applies smoothing** to metric trajectories using exponential moving averages
3. **Detects threshold crossings** when smoothed metrics reach target values
4. **Calculates costs** using real-time AWS pricing and SkyPilot instance information
5. **Logs results** to WandB for analysis and visualization

## Architecture

### Core Components

- **`PerformanceThresholdTracker`**: Main tracking class that monitors metrics and detects thresholds
- **`AWSPricingClient`**: Queries AWS Pricing API for real-time instance pricing
- **`SkyPilotInstanceInfo`**: Extracts instance information from SkyPilot environment
- **Configuration System**: YAML-based configuration for different environments

### Key Features

- **Real-time AWS Pricing**: Queries AWS Pricing API for up-to-date instance costs
- **SkyPilot Integration**: Automatically detects instance type, spot usage, and node configuration
- **Multi-node Support**: Calculates costs for distributed training across multiple nodes
- **Fallback Pricing**: Uses hardcoded pricing data when AWS API is unavailable
- **Environment Detection**: Automatically detects environment type from curriculum path

## Configuration

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
  instance_type: "g5.4xlarge"  # Default fallback
  use_spot: false  # Default fallback
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

## AWS Pricing System

### Real-time Pricing

The system queries the AWS Pricing API to get current instance pricing:

- **On-demand pricing**: Real-time on-demand instance costs
- **Spot pricing**: Real-time spot instance costs
- **Regional pricing**: Supports different AWS regions
- **Fallback data**: Hardcoded pricing when API is unavailable

### Instance Detection

The system automatically detects instance information from multiple sources:

1. **SkyPilot Environment Variables**:
   - `SKYPILOT_INSTANCE_TYPE`: Instance type (e.g., "g5.4xlarge")
   - `SKYPILOT_USE_SPOT`: Whether using spot instances
   - `SKYPILOT_NUM_NODES`: Number of nodes
   - `SKYPILOT_NUM_GPUS_PER_NODE`: GPUs per node

2. **SkyPilot Task YAML**: Parses task configuration files
3. **AWS Instance Metadata**: Queries instance metadata when running on AWS
4. **Configuration Fallback**: Uses default values from config files

### Cost Calculation

Cost calculation includes:

- **Instance pricing**: Real-time AWS pricing for the detected instance type
- **Spot vs on-demand**: Different pricing based on instance type
- **Multi-node scaling**: Cost multiplied by number of nodes
- **Time tracking**: Accurate elapsed time from training timer

Example cost calculation:
```
Cost = hours × price_per_hour × num_nodes
     = 2.5h × $2.424/h × 2 nodes = $12.12
```

## Supported Instance Types

The system supports pricing for common GPU instance types:

- **g4dn.xlarge**: 1 GPU, on-demand $0.526/h, spot $0.158/h
- **g5.xlarge**: 1 GPU, on-demand $1.006/h, spot $0.302/h
- **g5.2xlarge**: 1 GPU, on-demand $1.212/h, spot $0.364/h
- **g5.4xlarge**: 1 GPU, on-demand $2.424/h, spot $0.727/h
- **g5.8xlarge**: 1 GPU, on-demand $4.848/h, spot $1.454/h
- **g5.12xlarge**: 4 GPU, on-demand $7.272/h, spot $2.182/h
- **g5.24xlarge**: 8 GPU, on-demand $14.544/h, spot $4.363/h
- **p3.2xlarge**: 1 GPU, on-demand $3.06/h, spot $0.918/h
- **p3.8xlarge**: 4 GPU, on-demand $12.24/h, spot $3.672/h
- **p3.16xlarge**: 8 GPU, on-demand $24.48/h, spot $7.344/h

## Usage Examples

### Basic Training with Performance Tracking

```bash
# Run arena training with performance tracking
./recipes/arena_with_performance_tracking.sh
```

### Multi-node Training

```bash
# Launch multi-node training with performance tracking
./devops/skypilot/launch.py train run=multi_node_test --nodes 2 --gpus 4
```

### Custom Instance Types

```bash
# Use specific instance type with performance tracking
./devops/skypilot/launch.py train run=custom_instance --gpus 8
```

## Visualization

In WandB, you can create visualizations showing:

1. **Threshold Achievement Timeline**: When each threshold was reached
2. **Cost Efficiency**: Cost vs. performance comparison
3. **Training Efficiency**: Samples/time needed to reach thresholds
4. **Instance Utilization**: Cost per hour across different instance types

## Example WandB Queries

```python
# Get all performance threshold metrics
wandb.log({"performance_threshold/heart_gained_2/samples_to_threshold": samples})

# Check if threshold was achieved
if wandb.run.summary.get("performance_threshold/heart_gained_2/achieved"):
    print("Heart gained 2 threshold achieved!")

# Compare cost efficiency across runs
wandb.log({"performance_threshold/heart_gained_5/cost_to_threshold": cost})
```

## Implementation Details

### Core Components

- `PerformanceThresholdTracker`: Main tracking class
- `PerformanceThreshold`: Configuration dataclass
- `ThresholdResult`: Result dataclass
- `AWSPricingClient`: AWS pricing API client
- `SkyPilotInstanceInfo`: SkyPilot instance detection

### Key Methods

- `update()`: Update tracker with new metrics and check thresholds
- `get_results()`: Get current threshold results
- `get_wandb_metrics()`: Convert results to WandB format
- `calculate_total_cost()`: Calculate total cost for training run

### Error Handling

- **AWS API failures**: Falls back to hardcoded pricing data
- **Instance detection failures**: Uses default instance type
- **Metric validation**: Warns about unavailable metrics
- **Configuration errors**: Uses default arena thresholds

## Testing

Run the test script to verify the pricing system:

```bash
python test_aws_pricing.py
```

This will test:
- AWS pricing client functionality
- SkyPilot instance information extraction
- Cost calculation with different scenarios
- Performance threshold tracker integration

## Troubleshooting

### Common Issues

1. **AWS credentials not configured**: System will use fallback pricing
2. **Instance type not found**: Will use g5.4xlarge as default
3. **Metrics not available**: Check environment configuration
4. **Cost calculation errors**: Verify instance information

### Debug Information

Enable debug logging to see detailed information:

```python
import logging
logging.getLogger("metta.eval.aws_pricing").setLevel(logging.DEBUG)
logging.getLogger("metta.eval.performance_threshold_tracker").setLevel(logging.DEBUG)
```

## Future Enhancements

- **Additional cloud providers**: Support for GCP, Azure pricing
- **Reserved instance pricing**: Support for reserved instance discounts
- **Network costs**: Include data transfer and storage costs
- **GPU utilization**: Track actual GPU utilization for cost optimization
- **Automated cost alerts**: Notify when cost thresholds are exceeded
