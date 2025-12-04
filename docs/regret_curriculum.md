# Regret-Based Curriculum Learning

Implementation of regret-based curriculum learning inspired by the ACCEL paper:
**"Adversarially Compounding Complexity by Editing Levels"**

Reference: https://accelagent.github.io/

## Overview

This implementation provides two complementary regret-based curriculum strategies:

1. **PrioritizedRegret**: Prioritize tasks with highest regret (furthest from optimal)
2. **RegretLearningProgress**: Prioritize tasks where regret is decreasing fastest

Both approaches maintain tasks at the frontier of agent capabilities, following the ACCEL principle of producing levels at the boundary of what the agent can handle.

## What is Regret?

In curriculum learning, **regret** is defined as the gap between optimal performance and achieved performance:

```
regret = optimal_value - achieved_score
```

- **High regret** = Agent is far from optimal performance (task is challenging)
- **Low regret** = Agent is close to optimal performance (task is nearly solved)
- **Zero regret** = Agent achieves optimal performance (task is solved)

## Algorithms

### 1. PrioritizedRegret

**Strategy**: "Go to tasks with highest regret"

Prioritizes tasks where the agent is furthest from optimal performance, maintaining a challenging curriculum.

**Key characteristics**:
- Focuses on absolute regret values
- Keeps agent working on hardest unsolved tasks
- Good for maintaining challenge and preventing forgetting
- Similar to "frontier" task selection

**Use when**:
- You want to maintain agent capability on difficult tasks
- Preventing catastrophic forgetting is important
- You want a curriculum that challenges the agent consistently

**Configuration**:
```python
from metta.cogworks.curriculum.prioritized_regret_algorithm import PrioritizedRegretConfig

config = PrioritizedRegretConfig(
    optimal_value=1.0,              # Maximum achievable score
    regret_ema_timescale=0.01,      # EMA decay rate for regret
    exploration_bonus=0.1,          # Bonus for unexplored tasks
    temperature=1.0,                # Softmax temperature (higher = more random)
    min_samples_for_prioritization=2,  # Min samples before using regret
    max_memory_tasks=1000,          # Maximum tasks to track
)
```

### 2. RegretLearningProgress

**Strategy**: "Go to tasks where regret is getting lower fastest"

Prioritizes tasks where the agent is learning most rapidly, as measured by the rate of regret decrease.

**Key characteristics**:
- Focuses on rate of change of regret (learning rate)
- Identifies tasks at the "learning frontier"
- Good for accelerating learning on productive tasks
- Uses bidirectional EMAs (fast/slow) to detect learning

**Use when**:
- You want to accelerate learning on tasks where agent is improving
- You want to focus on productive learning opportunities
- You want automatic detection of learning plateaus

**Configuration**:
```python
from metta.cogworks.curriculum.regret_learning_progress_algorithm import RegretLearningProgressConfig

config = RegretLearningProgressConfig(
    optimal_value=1.0,              # Maximum achievable score
    regret_ema_timescale=0.001,     # EMA decay rate for regret
    use_bidirectional=True,         # Use fast/slow EMA comparison
    exploration_bonus=0.1,          # Bonus for unexplored tasks
    progress_smoothing=0.05,        # Smoothing for progress calculation
    invert_regret_progress=True,    # Prioritize decreasing regret
    min_samples_for_lp=2,           # Min samples before computing learning progress
    max_memory_tasks=1000,          # Maximum tasks to track
)
```

## Usage Examples

### Quick Start

```python
import metta.cogworks.curriculum as cc
from metta.cogworks.curriculum.prioritized_regret_algorithm import PrioritizedRegretConfig

# Create your environment
arena_env = make_arena_env()

# Create task variations
arena_tasks = cc.bucketed(arena_env)
arena_tasks.add_bucket("game.agent.rewards.inventory.laser", [0, 0.5, 1.0])

# Create curriculum with PrioritizedRegret
curriculum = arena_tasks.to_curriculum(
    algorithm_config=PrioritizedRegretConfig(
        optimal_value=1.0,
        temperature=1.0,
    )
)
```

### Using RegretLearningProgress

```python
from metta.cogworks.curriculum.regret_learning_progress_algorithm import RegretLearningProgressConfig

# Create curriculum with RegretLearningProgress
curriculum = arena_tasks.to_curriculum(
    algorithm_config=RegretLearningProgressConfig(
        optimal_value=1.0,
        use_bidirectional=True,
        invert_regret_progress=True,
    )
)
```

### Complete Training Example

See `recipes/experiment/regret_examples.py` for complete runnable examples:

```bash
# Train with PrioritizedRegret
./tools/run.py recipes.experiment.regret_examples.train_prioritized_regret

# Train with RegretLearningProgress
./tools/run.py recipes.experiment.regret_examples.train_regret_learning_progress
```

## Comparison with Standard Learning Progress

| Aspect | Learning Progress | PrioritizedRegret | RegretLearningProgress |
|--------|------------------|-------------------|------------------------|
| **Signal** | Success rate variance | Absolute regret | Regret change rate |
| **Focus** | Tasks with changing performance | Tasks far from optimal | Tasks improving fastest |
| **Strength** | General purpose, stable | Maintains challenge | Accelerates learning |
| **Use case** | Default choice | Hard task retention | Rapid skill acquisition |

## Key Concepts

### Regret vs Success Rate

- **Success Rate**: Absolute performance (e.g., 70% win rate)
- **Regret**: Distance from optimal (e.g., 30% below optimal)

Regret provides a normalized measure that works across tasks with different difficulty levels.

### Bidirectional EMAs

Both algorithms use fast and slow exponential moving averages:

- **Fast EMA**: Tracks recent performance (responds quickly to changes)
- **Slow EMA**: Tracks long-term trends (provides stable baseline)
- **Difference**: Captures rate of change (learning or forgetting)

### Task Eviction

Both algorithms implement smart eviction policies:

- **PrioritizedRegret**: Evicts tasks with lowest regret (nearly solved tasks)
- **RegretLearningProgress**: Evicts tasks with lowest learning progress (plateaued tasks)

This ensures the curriculum stays focused on productive learning opportunities.

## Monitoring and Logging

Both algorithms provide comprehensive statistics:

### PrioritizedRegret Stats
```python
stats = algorithm.stats()
# regret/mean_regret - Average regret across all tasks
# regret/total_tracked_tasks - Number of tasks being tracked
# regret/num_high_regret_tasks - Count of high-regret tasks
# regret/num_low_regret_tasks - Count of low-regret tasks
# regret/regret_std - Standard deviation of regret
```

### RegretLearningProgress Stats
```python
stats = algorithm.stats()
# regret_lp/num_tracked_tasks - Number of tasks being tracked
# regret_lp/mean_regret_lp - Average learning progress
# regret_lp/num_decreasing_regret - Tasks improving
# regret_lp/num_increasing_regret - Tasks degrading
# regret/mean_regret - Average regret
```

## Advanced Usage

### Custom Optimal Values

If your tasks don't have scores in [0, 1], adjust the optimal value:

```python
PrioritizedRegretConfig(
    optimal_value=100.0,  # For tasks scored 0-100
    ...
)
```

### Temperature Tuning

Control exploration-exploitation trade-off:

```python
PrioritizedRegretConfig(
    temperature=0.5,   # More greedy (focus on highest regret)
    # OR
    temperature=2.0,   # More exploratory (sample more uniformly)
)
```

### Regret EMA Timescale

Control how quickly regret estimates adapt:

```python
RegretLearningProgressConfig(
    regret_ema_timescale=0.001,  # Slower adaptation (stable)
    # OR
    regret_ema_timescale=0.1,    # Faster adaptation (responsive)
)
```

## Implementation Details

### RegretTracker

Core component that tracks regret for each task:

- Computes regret from scores: `regret = optimal - achieved`
- Maintains EMAs of regret values
- Tracks regret progress (fast vs slow EMA)
- Provides task statistics

### Algorithm Architecture

Both algorithms inherit from `CurriculumAlgorithm` and implement:

1. `score_tasks(task_ids)`: Return scores for task selection
2. `recommend_eviction(task_ids)`: Suggest which task to evict
3. `should_evict_task(task_id)`: Check if task meets eviction criteria
4. `update_task_performance(task_id, score)`: Update with new performance data

## Testing

Comprehensive tests are available in `tests/cogworks/curriculum/test_regret_algorithms.py`:

```bash
# Run all regret algorithm tests
pytest tests/cogworks/curriculum/test_regret_algorithms.py

# Run specific test class
pytest tests/cogworks/curriculum/test_regret_algorithms.py::TestPrioritizedRegretAlgorithm

# Run specific test
pytest tests/cogworks/curriculum/test_regret_algorithms.py::TestRegretTracker::test_regret_computation
```

## References

1. **ACCEL Paper**: "Adversarially Compounding Complexity by Editing Levels"
   - Interactive version: https://accelagent.github.io/
   - Key insight: Maintain tasks at the frontier of agent capabilities

2. **Related Work**:
   - PLR (Prioritized Level Replay)
   - PAIRED (Protagonist Antagonist Induced Regret Environment Design)
   - Teacher-Student Curriculum Learning

## FAQ

**Q: When should I use PrioritizedRegret vs RegretLearningProgress?**

A: Use PrioritizedRegret when you want to maintain challenge and prevent forgetting. Use RegretLearningProgress when you want to accelerate learning by focusing on tasks where the agent is improving fastest.

**Q: Can I use both algorithms together?**

A: Not directly in the same curriculum, but you could run separate experiments and ensemble the results, or switch between them during training.

**Q: How does this compare to standard LearningProgress?**

A: Regret-based approaches use the gap from optimal (regret) instead of raw success rates. This can be more robust across tasks with different difficulty levels and provides a clearer signal of "how far from solved" each task is.

**Q: What if my task doesn't have a clear optimal value?**

A: You can set `optimal_value` to the maximum achievable score in your domain, or use a learned estimate (e.g., best observed performance across all agents).

**Q: How sensitive are these algorithms to hyperparameters?**

A: The default values work well for most cases. The main parameter to tune is `regret_ema_timescale` - lower values (0.001) for stable, long-term learning; higher values (0.01-0.1) for faster adaptation to changing performance.

