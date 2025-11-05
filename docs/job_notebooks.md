# Job Notebooks

Generate Jupyter notebooks with WandB visualizations for training job groups.

## Quick Start

```bash
# 1. Launch training jobs (they will be part of a group)
uv run ./tools/run.py arena.train run=baseline trainer.total_timesteps=1000000
uv run ./tools/run.py arena.train run=high_lr trainer.total_timesteps=1000000

# 2. Monitor jobs
metta job monitor my_experiment

# 3. Generate notebook when done
metta job notebook my_experiment

# 4. Open notebook
jupyter notebook experiments/notebooks/my_experiment.ipynb
```

## Commands

### `metta job notebook <group>`

Generate a Jupyter notebook for all jobs in a group.

```bash
# Generate notebook with default path
metta job notebook my_experiment

# Custom output path
metta job notebook my_experiment --output ./analysis/results.ipynb
```

**Output**: `experiments/notebooks/<group>.ipynb` (gitignored)

### `metta job monitor <group>`

Monitor all jobs in a group with live updates.

```bash
metta job monitor my_experiment
```

### `metta job kill <group>`

Cancel all active jobs in a group.

```bash
metta job kill my_experiment
```

### `metta job list`

List all job groups and their status.

```bash
metta job list
```

## Generated Notebooks

Notebooks include 6 cells:

1. **Title**: Experiment name and run list
2. **Setup**: Imports and WandB configuration
3. **WandB URLs**: Clickable links to each run
4. **Fetch Metrics**: Uses `fetch_metrics()` from `experiments.notebooks.utils.metrics`
5. **Graphs**: Side-by-side reward and SPS plots
6. **Summary**: Statistics table

The notebooks use:
- Existing `experiments/notebooks/utils/metrics.py` for WandB fetching
- Sampled data (1000 points) for fast loading
- Matplotlib for visualization

## Architecture

- **Generation Code**: `metta/jobs/notebook_generation.py` (part of jobs system)
- **Notebook Output**: `experiments/notebooks/*.ipynb` (gitignored)
- **Shared Utilities**: `experiments/notebooks/utils/` (metrics, replays, etc.)

## Using with ExperimentTool

The ExperimentTool can optionally generate notebooks automatically:

```python
# In a recipe
def experiment() -> ExperimentTool:
    return ExperimentTool(
        module="arena.train",
        runs=["baseline", "variant"],
        group="my_experiment",
        generate_notebook=True,  # Auto-generate when complete
    )
```

Or generate manually after the fact:

```python
def experiment() -> ExperimentTool:
    return ExperimentTool(
        module="arena.train",
        runs=["baseline", "variant"],
        group="my_experiment",
        generate_notebook=False,  # Don't auto-generate
    )
```

```bash
# Launch experiment
uv run ./tools/run.py arena.experiment

# Generate notebook later
metta job notebook my_experiment
```

## Example Workflow

```bash
# Launch multiple training runs
for lr in 0.0001 0.0003 0.001; do
    uv run ./tools/run.py arena.train \
        run=lr_${lr}_$(date +%Y%m%d) \
        trainer.learning_rate=$lr \
        trainer.total_timesteps=5000000 \
        group=lr_sweep_$(date +%Y%m%d) &
done

# Monitor progress
metta job monitor lr_sweep_20250104

# Generate notebook when complete
metta job notebook lr_sweep_20250104

# Open and analyze
jupyter notebook experiments/notebooks/lr_sweep_20250104.ipynb
```

## Notes

- Notebooks are gitignored (`.gitignore` includes `experiments/notebooks/*.ipynb`)
- Notebook generation code is part of the jobs system (`metta/jobs/`)
- Shared utilities remain in `experiments/notebooks/utils/`
- Generated notebooks use existing utilities for consistency
