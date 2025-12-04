# Regret-Based Curriculum Learning Implementation

## Summary

This implementation adds regret-based curriculum learning to Metta, inspired by the ACCEL paper: "Adversarially Compounding Complexity by Editing Levels" (https://accelagent.github.io/).

## What Was Implemented

### 1. Core Infrastructure

**RegretTracker** (`metta/cogworks/curriculum/regret_tracker.py`)
- Computes regret as `optimal_value - achieved_score`
- Tracks regret EMAs (exponential moving averages)
- Monitors regret progress (fast vs slow EMA)
- Provides task statistics

### 2. Two Complementary Algorithms

**PrioritizedRegret** (`metta/cogworks/curriculum/prioritized_regret_algorithm.py`)
- **Strategy**: "Go to tasks with highest regret"
- Prioritizes tasks furthest from optimal performance
- Maintains challenging curriculum
- Good for preventing forgetting and maintaining agent capability

**RegretLearningProgress** (`metta/cogworks/curriculum/regret_learning_progress_algorithm.py`)
- **Strategy**: "Go to tasks where regret is getting lower fastest"
- Prioritizes tasks with fastest learning (decreasing regret)
- Identifies tasks at the "learning frontier"
- Good for accelerating learning on productive tasks

## Key Features

### Regret-Based Task Selection
- Uses gap from optimal (regret) instead of raw success rates
- More robust across tasks with different difficulty levels
- Clear signal of "how far from solved" each task is

### Bidirectional Learning Progress
- Fast EMA: Tracks recent performance changes
- Slow EMA: Provides stable baseline
- Difference captures learning rate (improving vs plateauing)

### Smart Task Eviction
- PrioritizedRegret: Evicts nearly-solved tasks (low regret)
- RegretLearningProgress: Evicts plateaued tasks (no progress)

### Comprehensive Statistics
Both algorithms provide detailed metrics:
- Mean regret across tasks
- Learning progress metrics
- Task distribution statistics
- Regret change rates

## Files Created/Modified

### New Files
```
metta/cogworks/curriculum/
├── regret_tracker.py                        (RegretTracker core component)
├── prioritized_regret_algorithm.py          (PrioritizedRegret algorithm)
├── regret_learning_progress_algorithm.py    (RegretLearningProgress algorithm)
└── README_REGRET.md                         (Quick reference)

recipes/experiment/
└── regret_examples.py                       (Example training recipes)

tests/cogworks/curriculum/
└── test_regret_algorithms.py                (Comprehensive tests)

docs/
└── regret_curriculum.md                     (Full documentation)
```

### Modified Files
```
metta/cogworks/curriculum/
├── __init__.py                              (Added exports)
└── curriculum.py                            (Added config types)
```

## Usage Examples

### Basic Usage (PrioritizedRegret)
```python
from metta.cogworks.curriculum.prioritized_regret_algorithm import PrioritizedRegretConfig
import metta.cogworks.curriculum as cc

arena_tasks = cc.bucketed(arena_env)
arena_tasks.add_bucket("game.agent.rewards.inventory.laser", [0, 0.5, 1.0])

curriculum = arena_tasks.to_curriculum(
    algorithm_config=PrioritizedRegretConfig(
        optimal_value=1.0,
        temperature=1.0,
    )
)
```

### Basic Usage (RegretLearningProgress)
```python
from metta.cogworks.curriculum.regret_learning_progress_algorithm import RegretLearningProgressConfig

curriculum = arena_tasks.to_curriculum(
    algorithm_config=RegretLearningProgressConfig(
        optimal_value=1.0,
        use_bidirectional=True,
        invert_regret_progress=True,
    )
)
```

### Running Examples
```bash
# Train with PrioritizedRegret
./tools/run.py recipes.experiment.regret_examples.train_prioritized_regret

# Train with RegretLearningProgress
./tools/run.py recipes.experiment.regret_examples.train_regret_learning_progress

# Compare all three approaches (LP, PR, RLP)
./tools/run.py recipes.experiment.regret_examples.compare_curricula
```

## Testing

Comprehensive test suite with 20+ tests covering:
- RegretTracker functionality
- PrioritizedRegret algorithm behavior
- RegretLearningProgress algorithm behavior
- Comparison between algorithms
- Edge cases and stress scenarios

```bash
# Run all regret tests
pytest tests/cogworks/curriculum/test_regret_algorithms.py

# Run specific test classes
pytest tests/cogworks/curriculum/test_regret_algorithms.py::TestRegretTracker
pytest tests/cogworks/curriculum/test_regret_algorithms.py::TestPrioritizedRegretAlgorithm
pytest tests/cogworks/curriculum/test_regret_algorithms.py::TestRegretLearningProgressAlgorithm
```

## Documentation

- **Full documentation**: `docs/regret_curriculum.md`
  - Detailed explanation of concepts
  - Configuration options
  - Advanced usage patterns
  - Comparison with standard learning progress
  - FAQ

- **Quick reference**: `metta/cogworks/curriculum/README_REGRET.md`
  - Quick start guide
  - File overview
  - Running examples

## Algorithm Comparison

| Aspect | Learning Progress | PrioritizedRegret | RegretLearningProgress |
|--------|------------------|-------------------|------------------------|
| **Signal** | Success rate variance | Absolute regret | Regret change rate |
| **Focus** | Changing performance | Far from optimal | Improving fastest |
| **Strength** | General purpose | Maintains challenge | Accelerates learning |
| **Use case** | Default | Hard task retention | Rapid skill acquisition |

## Key Hyperparameters

### PrioritizedRegret
- `optimal_value`: Maximum achievable score (default: 1.0)
- `temperature`: Softmax temperature for sampling (default: 1.0)
- `regret_ema_timescale`: EMA decay rate (default: 0.01)
- `exploration_bonus`: Bonus for unexplored tasks (default: 0.1)

### RegretLearningProgress
- `optimal_value`: Maximum achievable score (default: 1.0)
- `regret_ema_timescale`: EMA decay rate (default: 0.001)
- `use_bidirectional`: Use fast/slow EMA comparison (default: True)
- `invert_regret_progress`: Prioritize decreasing regret (default: True)
- `exploration_bonus`: Bonus for unexplored tasks (default: 0.1)

## Integration with Existing Code

The implementation seamlessly integrates with existing curriculum infrastructure:
- Inherits from `CurriculumAlgorithm` base class
- Uses existing `SliceAnalyzer` for task statistics
- Compatible with existing `Curriculum` and `CurriculumEnv`
- Works with existing task generators (`BucketedTaskGenerator`, etc.)

## Design Principles

1. **Minimal dependencies**: Uses only numpy beyond existing Metta infrastructure
2. **Consistent interface**: Follows same patterns as `LearningProgressAlgorithm`
3. **Comprehensive testing**: Full test coverage for core functionality
4. **Well documented**: Inline comments, docstrings, and external documentation
5. **Production ready**: Includes checkpointing, memory management, and error handling

## ACCEL Connection

This implementation realizes the ACCEL paper's core insights:
1. Use regret (distance from optimal) to identify frontier tasks
2. Maintain curriculum at the boundary of agent capabilities
3. Two complementary strategies:
   - Prioritize high absolute regret (challenging tasks)
   - Prioritize high regret reduction rate (productive learning)

## Future Extensions

Potential enhancements (not implemented):
- Level editing/generation based on regret patterns
- Multi-agent regret tracking
- Adaptive optimal value estimation
- Regret-based experience replay integration
- Hybrid algorithms combining multiple signals

## Verification

All code passes:
- ✅ Linting (no errors)
- ✅ Type checking (Pydantic models)
- ✅ Unit tests (comprehensive coverage)
- ✅ Integration with existing curriculum system

## References

1. **ACCEL Paper**: "Adversarially Compounding Complexity by Editing Levels"
   - Interactive version: https://accelagent.github.io/
   - Key contribution: Regret-based curriculum at agent capability frontier

2. **Related Curriculum Learning Work**:
   - PLR (Prioritized Level Replay)
   - PAIRED (Protagonist Antagonist Induced Regret Environment Design)
   - Teacher-Student Curriculum Learning

## Branch Information

- **Branch**: `regret-based-curriculum`
- **Based on**: `main`
- **Status**: Ready for review and testing

