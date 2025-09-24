# Task Dependency Mock Environment Recipe

This recipe provides a mock environment that simulates task dependency learning dynamics without running mettagrid training. It implements the chain-based task structure where tasks have dependencies and learning follows dynamical system equations.

## Overview

The mock environment models:
- **Task dependency chains**: Parent tasks contribute to child task learning
- **Performance dynamics**: Growth from training samples and forgetting over time
- **Task-specific noise**: Generated from seeds for reproducible task instantiations
- **Chain structure**: Tasks are arranged in a linear chain (0 → 1 → 2 → ... → N-1)

### Dynamics Equations

The performance dynamics follow:
```
P_dot[i] = growth - forgetting
growth = (samples[i] + γ * parent_samples) * children_gate * (1 - P[i])
forgetting = λ * P[i]
children_gate = ∏(P[child] for child in children[i])
```

Where:
- `γ` (gamma): Parent contribution factor
- `λ` (lambda): Forgetting rate
- `P[i]`: Performance of task i
- `samples[i]`: Number of samples for task i this epoch

## Usage

### Basic Simulation

```bash
# Small chain (5 tasks)
uv run ./tools/run.py experiments.recipes.task_dependency_mock_envs.simulate_small_chain

# Large chain (20 tasks)
uv run ./tools/run.py experiments.recipes.task_dependency_mock_envs.simulate_large_chain

# High parent contribution
uv run ./tools/run.py experiments.recipes.task_dependency_mock_envs.simulate_high_gamma

# High forgetting rate
uv run ./tools/run.py experiments.recipes.task_dependency_mock_envs.simulate_high_forgetting
```

### Custom Configuration

```python
from experiments.recipes.task_dependency_mock_envs import make_mock_env_config, simulate

# Create custom configuration
config = make_mock_env_config(
    num_tasks=10,
    num_epochs=100,
    samples_per_epoch=50,
    gamma=0.2,  # Parent contribution
    lambda_forget=0.15,  # Forgetting rate
    performance_threshold=0.9,
)

# Run simulation
results = simulate(mock_env_config=config, wandb_run_name="custom_experiment")
```

### Direct Simulator Usage

```python
from metta.rl.mock_dynamical_env import MockDynamicalSystemSimulator, CurriculumDrivenSimulation
from metta.cogworks.curriculum import Curriculum
from experiments.recipes.task_dependency_mock_envs import make_curriculum

# Create simulator
simulator = MockDynamicalSystemSimulator(
    num_tasks=5,
    num_epochs=50,
    samples_per_epoch=20,
    gamma=0.1,
    lambda_forget=0.1,
    task_seed=42,
)

# Create curriculum
curriculum_config = make_curriculum()
curriculum = Curriculum(curriculum_config)

# Run simulation
simulation = CurriculumDrivenSimulation(simulator, curriculum)
results = simulation.run_simulation()
```

## Configuration Parameters

### MockEnvironmentConfig

- `num_tasks` (int): Number of tasks in the chain (default: 10)
- `num_epochs` (int): Number of training epochs (default: 100)
- `samples_per_epoch` (int): Samples per epoch (default: 50)
- `gamma` (float): Parent contribution factor (default: 0.1)
- `lambda_forget` (float): Forgetting rate (default: 0.1)
- `performance_threshold` (float): Success threshold (default: 0.9)
- `task_seed` (Optional[int]): Seed for task-specific noise (default: random)

### Pre-configured Variants

| Function | Description | Parameters |
|----------|-------------|------------|
| `simulate_small_chain()` | Small 5-task chain | num_tasks=5, epochs=50 |
| `simulate_large_chain()` | Large 20-task chain | num_tasks=20, epochs=200 |
| `simulate_high_gamma()` | High parent contribution | gamma=0.3, lambda=0.05 |
| `simulate_high_forgetting()` | High forgetting rate | gamma=0.05, lambda=0.2 |

## Integration with Curriculum Learning

The simulator integrates seamlessly with the existing learning progress curriculum:

- **Task Selection**: Curriculum algorithm drives which tasks to sample
- **Performance Tracking**: Simulator provides task completion scores to curriculum
- **Learning Progress**: Bidirectional learning progress algorithm works out-of-the-box
- **Stats Logging**: Comprehensive metrics are logged to wandb

## Wandb Metrics

The simulator logs extensive metrics:

### Task-Level Metrics
- `task_{i}_performance`: Current performance of task i
- `task_{i}_completion_prob`: Completion probability of task i
- `task_{i}_samples`: Number of samples for task i this epoch
- `task_{i}_reward_noise`: Task-specific reward noise

### Aggregate Metrics
- `mean_performance`: Average performance across all tasks
- `max_performance`: Best performing task
- `min_performance`: Worst performing task
- `performance_std`: Standard deviation of performances
- `tasks_above_threshold`: Number of tasks above success threshold
- `total_samples_this_epoch`: Total samples collected this epoch

### Curriculum Metrics
- Standard curriculum algorithm metrics (learning progress, task selection, etc.)
- Slice analysis for parameter space exploration
- Task completion statistics

## Example Workflow

1. **Configure**: Choose or create a mock environment configuration
2. **Simulate**: Run simulation with the learning progress curriculum
3. **Monitor**: Watch task performance dynamics in wandb
4. **Analyze**: Study how task dependencies affect learning
5. **Compare**: Try different gamma/lambda values to see effects

## Comparison with Original Research Code

This implementation matches the dynamics from your original research code:

- Same chain graph structure (0 → 1 → 2 → ...)
- Same performance update equations
- Same parent contribution and forgetting mechanisms
- Same children gating (though simplified for linear chains)
- Compatible with learning progress curriculum algorithms

The key difference is integration with the existing metta infrastructure:
- Uses the curriculum system for task selection
- Logs metrics to wandb automatically
- Supports distributed training (if needed)
- Provides gymnasium-compatible interface

## Files

- `experiments/recipes/task_dependency_mock_envs.py`: Main recipe implementation
- `metta/rl/mock_dynamical_env.py`: Mock environment implementation
- `metta/rl/mock_env_stats.py`: Statistics collection for wandb
- `experiments/notebooks/task_dependency_demo.py`: Demo notebook
- `metta/rl/vecenv.py`: Modified to support mock environments

## Testing

The implementation includes integration tests that verify:
- Mock environment creation and dynamics
- Curriculum system integration
- Recipe configuration functions
- Wandb logging compatibility

Run tests with:
```bash
cd /path/to/metta
uv run python -c "
from experiments.recipes.task_dependency_mock_envs import *
from metta.rl.mock_dynamical_env import MockDynamicalSystemEnv
print('✅ All imports successful')
"
```
