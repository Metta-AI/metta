# Metta Sweep System

The sweep system provides hyperparameter optimization using the Protein optimizer, fully integrated with the Tool pattern.

## Quick Start

Run a sweep experiment:

```bash
# Run a hyperparameter sweep for arena
uv run ./tools/run.py experiments.sweep_arena.sweep_optimizer \
  --args sweep_name=my_arena_sweep num_trials=20

# Override Protein settings
uv run ./tools/run.py experiments.sweep_arena.sweep_optimizer \
  --args sweep_name=my_sweep num_trials=50 \
  --overrides sweep.protein.settings.num_random_samples=100
```

## Architecture

The sweep system follows the Tool pattern with these key components:

### 1. **ProteinConfig** (`protein_config.py`)
Configures the Protein optimizer:
- `metric`: What to optimize (e.g., "arena", "navigation")
- `goal`: "maximize" or "minimize"
- `parameters`: Nested dict of parameters to optimize
- `settings`: Protein algorithm settings (GP parameters, search strategy)

### 2. **SweepConfig** (`sweep_config.py`)
Overall sweep configuration:
- `num_trials`: How many trials to run
- `protein`: ProteinConfig instance
- `evaluation_simulations`: Optional evaluation tasks

### 3. **SweepTool** (`../tools/sweep.py`)
Tool wrapper that:
- Takes a TrainTool factory function
- Manages the sweep loop
- Applies Protein suggestions to TrainTools
- Records results back to Protein

### 4. **Core Sweep Function** (`sweep.py`)
The main sweep logic that:
- Initializes Protein with previous observations from WandB
- Generates suggestions for each trial
- Creates and invokes TrainTool instances
- Evaluates policies and records metrics
- Updates Protein with results

## Configuration Pipeline

### Step 1: Define Parameters to Optimize

```python
from metta.sweep.protein_config import ParameterConfig, ProteinConfig

protein_config = ProteinConfig(
    metric="arena",  # Metric to optimize
    goal="maximize",
    parameters={
        "trainer": {
            "optimizer": {
                "learning_rate": ParameterConfig(
                    min=1e-5, 
                    max=1e-2, 
                    distribution="log_normal"
                ),
            },
            "ppo": {
                "clip_coef": ParameterConfig(
                    min=0.05, 
                    max=0.3, 
                    distribution="uniform"
                ),
            }
        }
    }
)
```

### Step 2: Create Sweep Experiment

```python
from experiments.arena import train as arena_train_factory
from metta.tools.sweep import SweepTool
from metta.sweep.sweep_config import SweepConfig

def sweep_optimizer(sweep_name: str, num_trials: int = 10) -> SweepTool:
    # Define what to optimize
    protein_config = ProteinConfig(...)
    
    # Configure the sweep
    sweep_config = SweepConfig(
        num_trials=num_trials,
        protein=protein_config,
    )
    
    # Factory that creates TrainTools
    def train_factory(run_name: str) -> TrainTool:
        return arena_train_factory(run_name)
    
    return SweepTool(
        sweep=sweep_config,
        sweep_name=sweep_name,
        train_tool_factory=train_factory,
    )
```

### Step 3: Run the Sweep

```bash
uv run ./tools/run.py experiments.sweep_arena.sweep_optimizer \
  --args sweep_name=my_sweep num_trials=20
```

## How It Works

1. **Protein Optimization**: Uses Gaussian Process-based Bayesian optimization to explore the hyperparameter space efficiently

2. **Suggestion Generation**: For each trial, Protein generates a set of hyperparameters based on previous observations

3. **Training**: A TrainTool is created from the factory, suggestions are applied via the Tool's `override()` method, then training runs

4. **Evaluation**: After training, the policy is evaluated on specified simulations to compute the optimization metric

5. **Observation Recording**: Results (metric score, training cost) are recorded back to Protein and saved to WandB

6. **Persistence**: Observations are stored in WandB, allowing sweeps to be resumed or extended

## Parameter Distributions

Supported distributions for `ParameterConfig`:
- `"uniform"`: Linear uniform sampling
- `"log_normal"`: Log-scale sampling (good for learning rates)
- `"int_uniform"`: Integer uniform sampling

## Customization

### Custom Metrics
The metric can be any category from evaluation simulations (e.g., "navigation", "arena/combat") or "avg_category_score" for overall performance.

### Protein Settings
Fine-tune the optimizer via `ProteinSettings`:
- `num_random_samples`: Random samples before using GP
- `suggestions_per_pareto`: Suggestions per Pareto-optimal point
- `expansion_rate`: How quickly to expand search space

### Evaluation
Add custom evaluation simulations to compute domain-specific metrics:

```python
sweep_config = SweepConfig(
    evaluation_simulations=[
        SimulationConfig(name="arena/basic", ...),
        SimulationConfig(name="arena/combat", ...),
    ]
)
```

## Example: Complete Sweep Experiment

```python
# experiments/sweep_arena.py
from experiments.arena import train as arena_train_factory
from metta.sweep.protein_config import ParameterConfig, ProteinConfig
from metta.sweep.sweep_config import SweepConfig
from metta.tools.sweep import SweepTool

def sweep_optimizer(sweep_name: str, num_trials: int = 10) -> SweepTool:
    """Sweep for optimizing PPO hyperparameters on arena."""
    
    protein_config = ProteinConfig(
        metric="arena",
        goal="maximize",
        method="bayes",
        parameters={
            "trainer": {
                "optimizer": {
                    "learning_rate": ParameterConfig(
                        min=1e-5, max=1e-2, distribution="log_normal"
                    ),
                },
                "ppo": {
                    "clip_coef": ParameterConfig(
                        min=0.05, max=0.3, distribution="uniform"
                    ),
                    "ent_coef": ParameterConfig(
                        min=0.0001, max=0.01, distribution="log_normal"
                    ),
                    "gae_lambda": ParameterConfig(
                        min=0.8, max=0.99, distribution="uniform"
                    ),
                },
                "batch_size": ParameterConfig(
                    min=65536, max=1048576, distribution="int_uniform"
                ),
            }
        },
    )
    
    sweep_config = SweepConfig(
        num_trials=num_trials,
        protein=protein_config,
    )
    
    def train_factory(run_name: str) -> TrainTool:
        return arena_train_factory(run_name)
    
    return SweepTool(
        sweep=sweep_config,
        sweep_name=sweep_name,
        train_tool_factory=train_factory,
    )
```

## Distributed Training Note

The current implementation is designed for single-node sweeps. Key areas marked with `TODO: Adapt for distributed setup`:
- Protein suggestion generation (master only)
- WandB observation recording (master only)
- Evaluation execution (master only)

The underlying TrainTool handles distributed training automatically.

## Legacy System

The previous Hydra-based sweep system (`sweep_execute.py`, `sweep_lifecycle.py`) is being deprecated in favor of this Tool-pattern approach. Key improvements:
- Better composition with TrainTool
- Cleaner configuration with Pydantic
- No subprocess calls or intermediate YAML files
- Direct integration with the experiment system