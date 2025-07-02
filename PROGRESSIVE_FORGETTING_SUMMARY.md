# Progressive Forgetting Curriculum Package

## Overview

I've implemented a minimal but comprehensive package for measuring catastrophic forgetting in reinforcement learning agents through progressive curriculum training. The package enables systematic measurement of forgetting, transfer learning, and learning speed across all ordered pairs of the four task sets: Navigation, Memory, Navigation-Sequence, and Object Use (12 total experiments).

**Key Insight**: Order matters for catastrophic forgetting. Training on Navigation first then switching to Object Use will produce different forgetting patterns than training on Object Use first then switching to Navigation. The curriculum tests all 12 ordered pairs to capture these directional effects.

## Package Structure

```
metta/curriculum/
├── __init__.py                           # Package initialization
├── progressive_forgetting.py             # Main curriculum implementation
├── analysis.py                           # Analysis and visualization tools
└── README.md                             # Detailed documentation

configs/env/mettagrid/curriculum/
└── progressive_forgetting.yaml           # Curriculum configuration

configs/user/
└── progressive_forgetting.yaml           # User configuration

tools/
└── progressive_forgetting.py             # Main experiment runner

tests/curriculum/
└── test_progressive_forgetting.py        # Unit tests

examples/
└── progressive_forgetting_example.py     # Usage example
```

## Key Components

### 1. ProgressiveForgettingCurriculum

**Location**: `metta/curriculum/progressive_forgetting.py`

**Features**:
- Sharp switching between task sets when performance threshold is reached
- Performance tracking for all task sets throughout training
- Configurable parameters (threshold, smoothing, intervals)
- Random or deterministic task set ordering
- Comprehensive logging and statistics

**Key Methods**:
- `__init__()`: Initialize with task sets and parameters
- `complete_task()`: Update curriculum state after task completion
- `get_curriculum_stats()`: Return current curriculum statistics
- `_update_task_weights()`: Switch focus between task sets
- `_evaluate_all_task_sets()`: Track performance across all sets

### 2. ForgettingAnalyzer

**Location**: `metta/curriculum/analysis.py`

**Features**:
- Extract performance trajectories from training logs
- Calculate forgetting metrics for task set pairs
- Generate visualizations (matrices, plots)
- Comprehensive metric calculation

**Key Metrics Calculated**:
- **Zero-shot transfer**: Performance on task set 2 before training
- **Forgetting magnitude**: Peak performance minus final performance on task set 1
- **Learning magnitude**: Final performance minus zero-shot performance on task set 2
- **Learning speed**: Steps to reach 80% of final performance
- **Forgetting speed**: Steps to drop to 80% of peak performance

### 3. Main Experiment Runner

**Location**: `tools/progressive_forgetting.py`

**Features**:
- Run experiments for all ordered task set pairs (12 total)
- Automatic configuration generation for each pair
- Parallel experiment execution
- Comprehensive result analysis and reporting
- Visualization generation

## Configuration

### Task Sets

The curriculum is configured with four task sets, each containing 6 representative evaluation tasks:

1. **Navigation**: Basic navigation tasks (emptyspace, obstacles, mazes, etc.)
2. **Memory**: Memory-based tasks (easy, medium, hard memory challenges)
3. **Navigation Sequence**: Sequential navigation tasks
4. **Object Use**: Object interaction tasks (armory, generator, laser, etc.)

### Curriculum Parameters

```yaml
performance_threshold: 0.8    # Performance level to trigger switching
smoothing: 0.1               # Smoothing factor for performance tracking
switch_interval: 1000        # Minimum steps between switches
eval_interval: 100           # Steps between evaluations
randomize_order: true        # Randomize task set order
```

## Usage Examples

### Single Experiment

```bash
# Run a single progressive forgetting experiment
./tools/train.py run=test_forgetting +user=progressive_forgetting
```

### All Ordered Pair Experiments

```bash
# Run experiments for all ordered task set pairs (12 total)
./tools/progressive_forgetting.py --base-run-id=forgetting_study --total-timesteps=10000000
```

### Example Script

```bash
# Run the example demonstration
./examples/progressive_forgetting_example.py
```

## Output and Analysis

### Generated Files

1. **Raw Metrics**: `forgetting_metrics.csv` - All metrics for each task set pair
2. **Summary Statistics**: `summary_statistics.csv` - Mean, std, min, max for each metric
3. **Visualizations**:
   - `forgetting_matrix.png` - Forgetting magnitude between all pairs
   - `learning_matrix.png` - Learning magnitude between all pairs
   - `zero_shot_transfer_matrix.png` - Transfer learning between all pairs

### Example Results

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

## Integration with Existing Codebase

The package integrates seamlessly with the existing Metta infrastructure:

- **Curriculum System**: Extends the existing curriculum framework
- **Configuration**: Uses Hydra configuration system
- **Training**: Compatible with existing training pipeline
- **Evaluation**: Works with existing evaluation suites
- **Logging**: Integrates with existing logging and metrics systems

## Testing

The package includes comprehensive tests:

```bash
# Run unit tests
pytest tests/curriculum/test_progressive_forgetting.py -v

# Test imports
python -c "from metta.curriculum.progressive_forgetting import ProgressiveForgettingCurriculum"
python -c "from metta.curriculum.analysis import ForgettingAnalyzer"
```

## Key Features

### 1. Sharp Task Switching
- Abruptly switches from one task set to another when performance threshold is reached
- No gradual blending or mixing of tasks
- Clear separation for measuring forgetting

### 2. Comprehensive Metrics
- Measures forgetting magnitude and speed
- Quantifies transfer learning and learning speed
- Provides zero-shot performance baselines

### 3. Systematic Coverage
- Tests all ordered pairs of the four task sets (12 total experiments)
- Enables comparison of forgetting patterns across different task types and orders
- Identifies which task transitions and orders cause most forgetting

### 4. Visualization and Analysis
- Generates matrices showing forgetting patterns
- Creates performance trajectory plots
- Provides statistical summaries

### 5. Configurable and Extensible
- Easy to modify task sets and parameters
- Can be extended to additional task types
- Supports different performance thresholds and switching criteria

## Research Applications

This package enables systematic study of:

1. **Catastrophic Forgetting**: How much performance is lost when switching tasks
2. **Transfer Learning**: How much knowledge transfers between related tasks
3. **Learning Efficiency**: How quickly agents learn new tasks after forgetting
4. **Task Similarity**: Which task combinations show most/least forgetting
5. **Curriculum Design**: Optimal task ordering to minimize forgetting

## Future Extensions

The package can be extended to:

1. **Multi-task Sets**: Support for more than 2 task sets per experiment
2. **Adaptive Switching**: Dynamic switching based on learning progress
3. **Memory Mechanisms**: Integration with continual learning techniques
4. **Distributed Training**: Support for multi-GPU/multi-node experiments
5. **Real-time Analysis**: Live monitoring of forgetting during training

## Conclusion

This minimal package provides a complete solution for measuring catastrophic forgetting in reinforcement learning agents. It enables systematic experimentation across all task set combinations, provides comprehensive metrics and visualizations, and integrates seamlessly with the existing Metta codebase. The package is ready for immediate use and can be easily extended for more advanced research applications.
