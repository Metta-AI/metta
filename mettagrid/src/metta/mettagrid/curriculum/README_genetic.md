# GeneticBuckettedCurriculum

The `GeneticBuckettedCurriculum` is a curriculum learning approach that maintains a population of tasks and evolves them using genetic algorithms. Unlike the standard `BuckettedCurriculum` which generates all possible task combinations, this curriculum maintains a fixed-size population and continuously evolves it based on task performance.

## Key Features

- **Dynamic Population**: Maintains a fixed-size population of tasks instead of generating all combinations
- **Weighted Selection**: Tasks are selected for training based on their performance weights
- **Genetic Evolution**: Uses mutation and crossover operators to create new tasks
- **Adaptive Learning**: Automatically removes poorly-performing tasks and generates new ones
- **Parameter Constraints**: Ensures all generated tasks stay within defined parameter bounds

## How It Works

1. **Initialization**: Creates a random population of tasks within the defined parameter ranges
2. **Task Selection**: When `get_task()` is called, returns a task from the population based on weights
3. **Performance Tracking**: Updates task weights based on completion scores (exponential moving average)
4. **Evolution**: After each task completion:
   - Removes the bottom 10% (configurable) of tasks by weight
   - Generates new tasks using:
     - **Mutation** (30% by default): Modifies a single parameter using incr/decr/double/half operators
     - **Crossover** (70% by default): Combines parameters from two parent tasks
   - Parent selection is proportional to task weights

## Configuration

```yaml
_target_: metta.mettagrid.curriculum.genetic.GeneticBuckettedCurriculum

# Path to environment configuration template
env_cfg_template_path: /env/mettagrid/navigation/training/terrain_from_numpy

# Parameter specifications
buckets:
  # Discrete values
  game.map_builder.room.dir:
    values: ["terrain_maps_nohearts", "varied_terrain/balanced_large", ...]
  
  # Continuous ranges (integers)
  game.map_builder.room.width:
    range: [20, 64]
  
  # Continuous ranges (floats)
  game.objects.spawn_rate:
    range: [0.1, 1.0]

# Genetic algorithm parameters
population_size: 50      # Number of tasks in population
replacement_rate: 0.1    # Fraction of population to replace each generation
mutation_rate: 0.3       # Probability of mutation vs crossover
use_nevergrad: false     # Use nevergrad optimization if available

# Optional overrides
env_overrides:
  sampling: 0
```

## Parameters

- **population_size**: Size of the task population (default: 100)
- **replacement_rate**: Fraction of worst-performing tasks to replace each generation (default: 0.1)
- **mutation_rate**: Probability of using mutation vs crossover for new tasks (default: 0.3)
- **use_nevergrad**: Whether to use nevergrad for optimization if available (default: false)

## Genetic Operators

### Mutation
Selects a single parent and modifies one parameter:
- For discrete values: Randomly selects a different value
- For continuous values: Applies one of four operators:
  - **incr**: Increases by 10% of range
  - **decr**: Decreases by 10% of range
  - **double**: Doubles the value (clamped to range)
  - **half**: Halves the value (clamped to range)

### Crossover
Selects two parents and creates a child by randomly choosing each parameter from either parent.

## Usage Example

```python
from metta.api import Environment

# Create environment with genetic curriculum
env = Environment(
    curriculum_path="/env/mettagrid/curriculum/navigation/genetic",
    num_envs=32,
    num_agents=4,
)

# Training loop
for i in range(10000):
    obs = env.reset()
    done = False
    
    while not done:
        actions = policy(obs)
        obs, rewards, done, info = env.step(actions)
    
    # Task evolution happens automatically after completion
```

## Advantages

1. **Exploration**: Continuously explores new task combinations
2. **Adaptation**: Focuses on tasks that are neither too easy nor too hard
3. **Efficiency**: Maintains a smaller set of active tasks compared to exhaustive enumeration
4. **Flexibility**: Can handle large parameter spaces that would be intractable for BuckettedCurriculum

## Optional: Nevergrad Integration

If nevergrad is installed (`pip install nevergrad`), the curriculum can use more sophisticated optimization algorithms. Set `use_nevergrad: true` in the configuration to enable this feature.

## Tips

1. **Population Size**: Larger populations provide more diversity but require more memory
2. **Replacement Rate**: Higher rates lead to faster evolution but may be unstable
3. **Mutation Rate**: Balance between exploration (mutation) and exploitation (crossover)
4. **Parameter Ranges**: Ensure ranges are reasonable to avoid generating impossible tasks