# Curricula

Curricula control how environments are selected during training, allowing agents to learn progressively from easier to harder tasks.

They represent a space of possible environments (a set of tasks) and an algorithm for deciding how to sample between them over time.

A Curriculum consists of:
* **Curriculum tree**: A hierarchical structure of tasks (which may themselves be curricula containing subtasks)
* **CurriculumAlgorithm**: The strategy for adjusting sampling rates between tasks based on agent performance

## Core Components

### curriculum.py
Contains the base `Curriculum` and `MettaGridTask` classes. MettaGridEnv is the main client - it samples tasks during initialization and includes curriculum statistics in episode results.

### curriculum_builder.py
Provides helper functions to construct curricula:
- `single_task()`: Creates a curriculum with one task
- `task_set()`: Creates a flat curriculum from multiple tasks
- `parameter_grid_task_set()`: Generates tasks by varying parameters across a grid

Typically, we build maps where style and resource density are preserved within tasks, while terrain and placement vary randomly. You can also create highly specific tasks using parameter grids.

## Curriculum Algorithms

We currently support 4 algorithms, all using reward as their primary signal:

* **DiscreteRandom**: Static random sampling with configurable weights (defaults to uniform)
* **PrioritizeRegressed**: Prioritizes tasks where historical performance was much better than recent performance
* **LearningProgress**: Focuses on tasks where reward is improving fastest
* **Progressive**: Pre-sequenced tasks with progression gated by time or reward thresholds

Note: Using regret as an input signal is high on Jack's priority list for future exploration.

## Usage Example

```python
from metta.mettagrid.curriculum import task_set, DiscreteRandomHypers

# Create a simple curriculum with three tasks
curriculum = task_set(
    name="navigation_basics",
    env_configs=[
        ("easy", easy_config),
        ("medium", medium_config),
        ("hard", hard_config)
    ],
    curriculum_hypers=DiscreteRandomHypers(initial_weights=[0.5, 0.3, 0.2])
)
```