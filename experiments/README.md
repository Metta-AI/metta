# Experiments

A framework for reproducible experiments and research notebooks with Metta.

## Quick Start

### Research Notebooks (Empty Template)

```bash
# Create an empty research notebook
./experiments/new_notebook.py my_research

# With a description and custom sections
./experiments/new_notebook.py optimizer_study --description "Comparing optimizers" --sections setup,state,metrics,log
```

### Experiment Notebooks (Pre-configured with Runs)

```bash
# Create and launch from arena recipe
./experiments/new_notebook.py arena_test --recipe arena

# Create from recipe without launching
./experiments/new_notebook.py arena_analysis --recipe arena --no-launch

# Create with custom config
./experiments/new_notebook.py custom_exp --curriculum env/mettagrid/curriculum/test --gpus 2 --tags research,ablation
```

### Run Existing Experiments

```bash
# Run arena experiment directly
python experiments/recipes/arena_experiment.py
```

## Structure

```
experiments/
├── experiment.py          # Base Experiment class
├── launch.py             # Core training launch functionality
├── new_notebook.py       # Unified notebook creation script
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
└── log/                  # Generated notebooks and metadata
```

## Creating New Experiments

### Method 1: Using new_notebook.py (Recommended for Quick Iteration)

```bash
# Research notebook - empty template for exploration
./experiments/new_notebook.py my_research

# With pre-launched training runs
./experiments/new_notebook.py my_experiment --curriculum env/mettagrid/curriculum/test --gpus 4
```

### Method 2: Creating an Experiment Class (Recommended for Reproducibility)

Create a new file in `experiments/recipes/`:

```python
from experiments.experiment import Experiment
from experiments.types import TrainingJob, TrainingJobConfig

class MyExperiment(Experiment):
    def launch_training_runs(self) -> List[TrainingJob]:
        # Create config
        config = TrainingJobConfig(
            curriculum="env/mettagrid/my_curriculum",
            gpus=4,
            nodes=1,
            wandb_tags=["my_experiment"],
            additional_args=[
                "trainer.optimizer.learning_rate=0.001",
                "trainer.optimizer.type=adam"
            ]
        )
        
        # Launch using config
        job = self.launch_training_run_from_config("my_run_name", config)
        return [job] if job else []

    def get_analysis_config(self):
        return {
            "metrics_to_plot": ["overview/reward"],
            "eval_suites": ["navigation"],
        }
```

## Notebook Sections

Available sections (use `--sections` to customize):
- **setup**: Imports and configuration
- **state**: Run tracking and management
- **launch**: Training launch examples
- **monitor**: Status monitoring
- **metrics**: Metric fetching and analysis
- **visualize**: Plotting and visualizations
- **replays**: MettaScope replay viewer
- **log**: Experiment documentation
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
- The `experiments/log/` directory contains all generated notebooks
- Notebooks auto-save as you work in Jupyter