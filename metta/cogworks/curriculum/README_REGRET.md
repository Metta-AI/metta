# Regret-Based Curriculum Learning

This directory contains implementations of regret-based curriculum learning algorithms inspired by the ACCEL paper.

## Quick Reference

### Files

- **`regret_tracker.py`**: Core regret tracking component
- **`prioritized_regret_algorithm.py`**: PrioritizedRegret curriculum (highest regret)
- **`regret_learning_progress_algorithm.py`**: RegretLearningProgress curriculum (fastest regret decrease)

### Algorithms

1. **PrioritizedRegret**: Prioritize tasks with highest regret
   - Strategy: "Go to tasks with highest regret"
   - Good for: Maintaining challenge, preventing forgetting

2. **RegretLearningProgress**: Prioritize tasks where regret is decreasing fastest
   - Strategy: "Go to tasks where regret is getting lower fastest"
   - Good for: Accelerating learning, finding productive tasks

## Quick Start

```python
from metta.cogworks.curriculum.prioritized_regret_algorithm import PrioritizedRegretConfig

curriculum = arena_tasks.to_curriculum(
    algorithm_config=PrioritizedRegretConfig(
        optimal_value=1.0,
        temperature=1.0,
    )
)
```

## Examples

See:
- **Full documentation**: `/docs/regret_curriculum.md`
- **Example recipes**: `/recipes/experiment/regret_examples.py`
- **Tests**: `/tests/cogworks/curriculum/test_regret_algorithms.py`

## Running Examples

```bash
# Train with PrioritizedRegret
./tools/run.py recipes.experiment.regret_examples.train_prioritized_regret

# Train with RegretLearningProgress
./tools/run.py recipes.experiment.regret_examples.train_regret_learning_progress

# Compare all approaches
./tools/run.py recipes.experiment.regret_examples.compare_curricula
```

## Testing

```bash
# Run all regret tests
pytest tests/cogworks/curriculum/test_regret_algorithms.py

# Run specific algorithm tests
pytest tests/cogworks/curriculum/test_regret_algorithms.py::TestPrioritizedRegretAlgorithm
pytest tests/cogworks/curriculum/test_regret_algorithms.py::TestRegretLearningProgressAlgorithm
```

## Reference

Based on **"Adversarially Compounding Complexity by Editing Levels" (ACCEL)**
- Paper: https://accelagent.github.io/
- Key insight: Maintain tasks at the frontier of agent capabilities using regret signals

