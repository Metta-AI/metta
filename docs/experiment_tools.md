# Experiment Tools

This document describes the new experiment tools for launching and managing parallel training jobs.

## Overview

The experiment system provides two main capabilities:

1. **ExperimentTool**: A recipe tool for launching parallel training jobs and generating analysis notebooks
2. **Job Management Commands**: CLI commands for monitoring and controlling job groups

## ExperimentTool

The `ExperimentTool` allows you to launch multiple training jobs in parallel from a recipe, monitor their progress, and automatically generate a Jupyter notebook with reward and SPS graphs.

### Basic Usage

In your recipe file (e.g., `experiments/recipes/arena.py`):

```python
from metta.tools.experiment import ExperimentTool

def experiment(
    runs: list[str] | None = None,
    group: str | None = None,
    gpus: int = 4,
) -> ExperimentTool:
    """Launch a parallel experiment with multiple training runs."""
    if runs is None:
        runs = ["baseline", "variant"]

    return ExperimentTool(
        module="arena.train",  # Tool to run for each job
        runs=runs,             # List of run names
        args_per_run={         # Optional: per-run arguments
            "baseline": ["trainer.total_timesteps=1000000"],
            "variant": ["trainer.total_timesteps=1000000", "trainer.learning_rate=0.001"],
        },
        group=group,           # Job group name (auto-generated if None)
        gpus=gpus,             # GPUs per job
        generate_notebook=True,  # Generate analysis notebook
    )
```

### Launching an Experiment

```bash
# Launch with default runs
uv run ./tools/run.py arena.experiment

# Launch with custom runs
uv run ./tools/run.py arena.experiment \
    runs='["baseline","high_lr","low_lr"]' \
    group=my_experiment

# Launch with custom GPU count
uv run ./tools/run.py arena.experiment \
    runs='["run1","run2"]' \
    gpus=8
```

### Features

- **Parallel Job Execution**: All runs are submitted simultaneously and execute in parallel
- **Live Monitoring**: Shows real-time progress of all jobs
- **Automatic Notebook Generation**: Creates a Jupyter notebook with:
  - WandB run URLs
  - Agent reward graphs over time
  - Steps-per-second (SPS) graphs
  - Summary statistics table
- **Failure Detection**: Detects and reports failed jobs

### Generated Notebook

The notebook is saved to `experiments/notebooks/<group>.ipynb` (gitignored) and includes:

1. **Title cell**: Experiment name and run list
2. **Setup cell**: Imports `fetch_metrics` from `experiments.notebooks.utils.metrics`
3. **WandB URLs**: Direct links to each run (clickable)
4. **Fetch metrics cell**: Uses shared `fetch_metrics()` utility to load data from WandB
5. **Visualization cell**: Side-by-side reward and SPS plots using matplotlib
6. **Summary cell**: Statistics table with final/max/mean rewards and average SPS

The notebook uses the existing `experiments/notebooks/utils/metrics.py` module for fetching data from WandB, which provides:
- Sampled data (1000 points by default) for fast loading
- Proper error handling and progress reporting
- Shared code with other notebook utilities

## Job Management Commands

The `metta job` commands allow you to monitor and control job groups from the CLI.

### List Job Groups

```bash
metta job list
```

Shows all job groups with counts by status:
- Total jobs
- Running jobs
- Pending jobs
- Completed jobs
- Failed jobs

### Monitor Jobs

```bash
# Monitor a specific group
metta job monitor my_experiment

# Monitor with custom refresh interval
metta job monitor my_experiment --refresh 2.0

# Monitor without showing logs
metta job monitor my_experiment --no-logs
```

The monitor displays:
- **Failed jobs** at the top for visibility
- **Active jobs** with progress bars and live log tails
- **Completed jobs** with artifacts (WandB URLs, checkpoints)
- **Pending jobs** with dependency information

Press Ctrl+C to exit (jobs continue running).

### Kill Jobs

```bash
# Kill all jobs in a group (with confirmation)
metta job kill my_experiment

# Kill without confirmation
metta job kill my_experiment --force
```

This will:
- Terminate all running jobs
- Cancel all pending jobs
- Leave completed jobs unchanged

## Complete Workflow Example

```bash
# 1. Launch an experiment with multiple runs
uv run ./tools/run.py arena.experiment \
    runs='["baseline","high_lr","low_lr","high_entropy"]' \
    group=lr_sweep \
    gpus=4

# The experiment tool will:
# - Submit all 4 jobs to the job manager
# - Monitor their progress with live updates
# - Generate a notebook when complete

# 2. (Optional) In another terminal, monitor progress
metta job monitor lr_sweep

# 3. (Optional) If you need to cancel the experiment
metta job kill lr_sweep

# 4. View the generated notebook (note: notebooks are gitignored)
jupyter notebook experiments/notebooks/lr_sweep.ipynb
```

## Advanced Usage

### Custom Per-Run Arguments

```python
def experiment() -> ExperimentTool:
    return ExperimentTool(
        module="arena.train",
        runs=["baseline", "high_lr", "low_entropy"],
        args_per_run={
            "baseline": [],
            "high_lr": ["trainer.learning_rate=0.001"],
            "low_entropy": ["trainer.entropy_coef=0.001"],
        },
        group="custom_args_experiment",
    )
```

### Local vs Remote Execution

By default, jobs run remotely via SkyPilot. To run locally:

```python
def experiment() -> ExperimentTool:
    return ExperimentTool(
        module="arena.train",
        runs=["test_run"],
        remote=False,  # Run locally instead of via SkyPilot
    )
```

### Custom Metrics Tracking

```python
def experiment() -> ExperimentTool:
    return ExperimentTool(
        module="arena.train",
        runs=["run1", "run2"],
        metrics_to_track=[
            "overview/sps",
            "overview/reward",
            "trainer/loss",
            "trainer/entropy",
        ],
    )
```

### Custom Notebook Path

```python
from pathlib import Path

def experiment() -> ExperimentTool:
    return ExperimentTool(
        module="arena.train",
        runs=["run1"],
        notebook_path=Path("./my_analysis/results.ipynb"),
    )
```

## Migration from Old experiment() Pattern

The old pattern in `assembly_lines.py`:

```python
def experiment():
    for curriculum_style in curriculum_args:
        subprocess.run([
            "./devops/skypilot/launch.py",
            "experiments.recipes.assembly_lines.train",
            f"run=assembly_lines_{curriculum_style}.{time.strftime('%Y-%m-%d')}",
            f"curriculum_style={curriculum_style}",
            "--gpus=4",
        ])
```

Can be replaced with:

```python
def experiment() -> ExperimentTool:
    runs = [f"assembly_lines_{style}" for style in curriculum_args.keys()]
    return ExperimentTool(
        module="assembly_lines.train",
        runs=runs,
        args_per_run={
            f"assembly_lines_{style}": [f"curriculum_style={style}"]
            for style in curriculum_args.keys()
        },
        gpus=4,
        generate_notebook=True,
    )
```

Benefits of the new approach:
- Live monitoring of all jobs
- Automatic failure detection
- Generated analysis notebook
- Better integration with job management system
- Easier to resume/monitor from CLI

## Architecture

### Job State Persistence

Jobs are persisted to `job_state/jobs.sqlite` by default. This allows:
- Resuming monitoring after CLI restart
- Querying historical job information
- Tracking dependencies between jobs

### Job Group Management

Jobs with the same `group` value are managed together:
- Monitor all jobs in a group simultaneously
- Cancel all jobs in a group with one command
- Generate notebooks for all jobs in a group

### Integration with Existing Systems

The experiment tools build on the existing metta/jobs system:
- Uses `JobManager` for orchestration
- Uses `JobDisplay` for live monitoring
- Uses `JobConfig` for job configuration
- Compatible with existing SkyPilot infrastructure
- Integrates with WandB for metrics tracking
