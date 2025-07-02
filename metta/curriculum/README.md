# Progressive Forgetting Curriculum

This package implements a progressive curriculum for measuring catastrophic forgetting in reinforcement learning agents. It trains agents on one task set until they reach a performance threshold, then sharply switches to another task set, allowing measurement of forgetting and transfer learning.

## Overview

The progressive forgetting curriculum is designed to:

1. **Measure Catastrophic Forgetting**: Track how much performance degrades on the first task set when training switches to a second task set
2. **Measure Transfer Learning**: Assess zero-shot performance on the second task set before training on it
3. **Measure Learning Speed**: Determine how quickly the agent learns the second task set
4. **Generate Comprehensive Metrics**: Provide matrices of forgetting, transfer, and learning metrics across all ordered task set pairs (12 total)

**Note**: Order matters for forgetting measurement. Training on Navigation first then switching to Object Use will produce different forgetting patterns than training on Object Use first then switching to Navigation. The curriculum tests all 12 ordered pairs to capture these directional effects.

## Components

### ProgressiveForgettingCurriculum

The main curriculum class that implements sharp switching between task sets:

- **Sharp Switching**: Abruptly switches from one task set to another when performance threshold is reached
- **Performance Tracking**: Monitors performance on all task sets throughout training
- **Configurable Parameters**: Adjustable thresholds, intervals, and smoothing factors

### ForgettingAnalyzer

Analysis tools for extracting and visualizing forgetting metrics:

- **Performance Trajectories**: Extract learning curves for each task set
- **Forgetting Metrics**: Calculate forgetting magnitude, transfer learning, and learning speed
- **Visualizations**: Generate matrices and plots of forgetting patterns

## Task Sets

The curriculum is configured with four main task sets:

1. **Navigation**: Basic navigation tasks (emptyspace, obstacles, mazes, etc.)
2. **Memory**: Memory-based tasks (easy, medium, hard memory challenges)
3. **Navigation Sequence**: Sequential navigation tasks
4. **Object Use**: Object interaction tasks (armory, generator, laser, etc.)

## Usage

### Running a Single Experiment

```bash
# Run a single progressive forgetting experiment
./tools/train.py run=test_forgetting +user=progressive_forgetting trainer.curriculum=/env/mettagrid/curriculum/progressive_forgetting
```

### Running All Ordered Pair Experiments

```bash
# Run experiments for all ordered task set pairs (12 total)
./tools/progressive_forgetting.py --base-run-id=forgetting_study --total-timesteps=10000000
```

### Command Line Options

- `--task-sets`: Specify which task sets to use (default: all four)
- `--base-run-id`: Base identifier for training runs
- `--total-timesteps`: Training timesteps per experiment
- `--num-workers`: Number of parallel workers
- `--device`: Training device (cpu/gpu)
- `--output-dir`: Directory to save results

## Configuration

### Curriculum Parameters

```yaml
# Progressive forgetting parameters
performance_threshold: 0.8    # Performance level to trigger task switching
smoothing: 0.1               # Smoothing factor for performance tracking
switch_interval: 1000        # Minimum steps between task switches
eval_interval: 100           # Steps between evaluations of all task sets
randomize_order: true        # Whether to randomize task set order
```

### Task Set Configuration

Each task set is defined as a dictionary mapping task paths to weights:

```yaml
task_sets:
  navigation:
    /env/mettagrid/navigation/evals/emptyspace_withinsight: 1
    /env/mettagrid/navigation/evals/obstacles1: 1
    # ... more navigation tasks

  memory:
    /env/mettagrid/memory/evals/easy: 1
    /env/mettagrid/memory/evals/medium: 1
    # ... more memory tasks
```

## Metrics

The analysis produces several key metrics for each task set pair:

### Forgetting Metrics

- **forgetting_magnitude**: Peak performance on task set 1 minus final performance after switching
- **forgetting_speed**: Steps to drop to 80% of peak performance on task set 1

### Transfer Learning Metrics

- **zero_shot_transfer**: Performance on task set 2 before training on it
- **learning_magnitude**: Final performance on task set 2 minus zero-shot performance
- **learning_speed**: Steps to reach 80% of final performance on task set 2

## Output

The experiments generate:

1. **Raw Metrics**: CSV file with all forgetting metrics for each pair
2. **Summary Statistics**: Mean, std, min, max for each metric type
3. **Visualizations**:
   - Forgetting magnitude matrix
   - Learning magnitude matrix
   - Zero-shot transfer matrix
   - Performance trajectory plots

## Example Results

After running all experiments, you'll get:

```
PROGRESSIVE FORGETTING EXPERIMENT SUMMARY
================================================================================

Total experiments: 12 (12 ordered pairs)
Successful experiments: 12

Summary Statistics:
  zero_shot_transfer:
    Mean: 0.2345
    Std:  0.1234
    Range: [0.1000, 0.4000]
  forgetting_magnitude:
    Mean: 0.3456
    Std:  0.2345
    Range: [0.2000, 0.6000]
  learning_magnitude:
    Mean: 0.4567
    Std:  0.3456
    Range: [0.3000, 0.7000]
```

## Testing

Run the tests to verify the curriculum works correctly:

```bash
pytest tests/curriculum/test_progressive_forgetting.py -v
```

## Integration

The progressive forgetting curriculum integrates with the existing Metta training infrastructure:

- Uses the same task configuration system
- Compatible with existing evaluation suites
- Logs metrics through the standard training pipeline
- Works with local and distributed training setups
