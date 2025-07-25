# Experiments

A framework for reproducible experiments and research notebooks with Metta.

## Quick Start

### Create and Launch Experiments

```bash
# Create and launch arena experiment
./experiments/recipes/arena_experiment.py

# Create notebook without launching (load existing jobs)
./experiments/recipes/arena_experiment.py --no-launch --job-ids 2979 2980 --open

# Launch with custom configuration
./experiments/recipes/arena_experiment.py my_arena --gpus 4 --skip-git-check --wandb-tags research ablation

# Create notebook with custom sections
./experiments/recipes/arena_experiment.py analysis --sections monitor,analysis,replays --open
```

## Structure

```
experiments/
├── experiment.py          # Base Experiment class
├── launch.py             # Core training launch functionality
├── types.py              # TrainingJob and TrainingJobConfig
├── wandb_utils.py        # WandB data fetching utilities
├── monitoring.py         # Sky job monitoring utilities
├── recipes/              # Experiment implementations
│   └── arena_experiment.py
├── notebooks/            # Notebook utilities
│   ├── generation.py     # Core notebook generation engine
│   ├── analysis.py       # Analysis and visualization functions
│   ├── monitoring.py     # Display widgets for monitoring
│   ├── replays.py        # MettaScope replay utilities
│   └── training.py       # Training launch helpers
├── scratch/              # Generated notebooks (git-ignored)
└── log/                  # HTML exports from notebooks
```

## Creating New Experiments

Each experiment is a standalone script in `experiments/recipes/`. To create a new experiment:

1. Copy an existing recipe (e.g., `arena_experiment.py`)
2. Define your experiment-specific config class
3. Implement the launch and analysis methods
4. Add command-line arguments for your config fields

Example structure:

```python
from experiments.experiment import Experiment
from experiments.types import TrainingJob, TrainingJobConfig, BaseExperimentConfig
from pydantic import Field
from typing import Optional, List

class MyExperimentConfig(BaseExperimentConfig):
    """Configuration for my experiment."""
    # Launch parameters
    curriculum: str = Field("env/mettagrid/my_curriculum", description="Curriculum path")
    gpus: int = Field(4, description="Number of GPUs")
    learning_rate: float = Field(0.001, description="Learning rate")
    
class MyExperiment(Experiment):
    def __init__(self, name: str, config: MyExperimentConfig):
        super().__init__(name)
        self.config = config
        
    def launch_training_runs(self) -> List[TrainingJob]:
        config = TrainingJobConfig(
            curriculum=self.config.curriculum,
            gpus=self.config.gpus,
            additional_args=[f"trainer.optimizer.learning_rate={self.config.learning_rate}"]
        )
        
        job = self.launch_training_run_from_config("my_run", config)
        return [job] if job else []
```

## Notebook Sections

Available sections (use `--sections` to customize):
- **setup**: Imports and configuration
- **state**: Run tracking and management
- **launch**: Training launch examples
- **monitor**: Status monitoring
- **analysis**: Analysis and visualizations (SPS plot)
- **replays**: MettaScope replay viewer
- **scratch**: Quick experiments

## Key Utilities

### Training
- `launch_training()` - Launch a single training run
- `launch_multiple_training_runs()` - Launch multiple runs with seed variation

### Monitoring
- `monitor_training_statuses()` - Check status with live metrics
- `find_training_jobs()` - Find runs by author, tags, state

### Analysis
- `fetch_metrics()` - Pull metrics from wandb
- `plot_sps()` - Plot steps per second
- `create_run_summary_table()` - Generate summary statistics

### Replays
- `show_replay()` - Display MettaScope replay
- `get_available_replays()` - List available replay steps

## Tips

- Use descriptive names with dates for tracking
- Document hypotheses in the experiment log section
- The `experiments/scratch/` directory contains all generated notebooks (git-ignored)
- The `experiments/log/` directory contains HTML exports
- Notebooks auto-save as you work in Jupyter