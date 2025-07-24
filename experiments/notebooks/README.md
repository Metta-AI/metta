# Experiments

A framework for reproducible experiments that auto-generate analysis notebooks.

## Quick Start

```bash
# Run an existing experiment
python experiments/templates/arena_experiment.py

```

This will:

1. Launch training runs on the cluster
2. Generate an analysis notebook in `experiments/log/`
3. Save experiment metadata for reproducibility

## Creating a New Experiment

Create a new file in `experiments/templates/`:

```python
from experiments.experiment import Experiment
from experiments.launch import launch_training_run

class MyExperiment(Experiment):
    def launch_training_runs(self):
        result = launch_training_run(
            run_name="my_run",
            curriculum="env/mettagrid/my_curriculum",
            num_gpus=1,
        )
        self.launch_results.append(result)
        return {
            "run_names": [result["run_name"]],
            "success": result["success"]
        }

    def get_analysis_config(self):
        return {
            "metrics_to_plot": ["overview/reward"],
            "eval_suites": ["navigation"],
        }
```

## Structure

```
experiments/
├── experiment.py          # Base Experiment class
├── launch.py             # Core training launch functionality
├── templates/            # Experiment implementations
│   └── arena_experiment.py
├── notebooks/            # Notebook utilities and analysis tools
└── log/                  # Generated notebooks and metadata
```

## Experiment Interface

All experiments inherit from `Experiment` and must implement:

- `launch_training_runs()`: Launch training and return metadata
- `get_analysis_config()`: Specify metrics and analysis configuration

Generated notebooks include:

- Experiment metadata and configuration
- Pre-configured analysis cells
- Links to training runs
- Standard visualizations (SPS, metrics, etc.)
