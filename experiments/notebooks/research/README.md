# Research Notebooks

This directory contains research notebooks for iterative experimentation with Metta.

## Creating a New Research Notebook

```bash
# From the metta root directory:
./experiments/new_research_notebook.py my_experiment

# With a description:
./experiments/new_research_notebook.py optimizer_study --description "Comparing different optimizers"

# Custom output directory:
./experiments/new_research_notebook.py ablation --output-dir experiments/notebooks/research/2024/
```

## What's Included

Each generated notebook includes:

- **Setup**: Auto-reload, imports, and configuration
- **Experiment Tracking**: Simple state management for multiple runs
- **Launch Training**: Examples for single and multiple training runs
- **Monitor Training**: Live status monitoring with wandb integration
- **Fetch & Analyze Metrics**: Tools to pull and analyze training metrics
- **Visualizations**: Pre-configured plotting for common metrics
- **View Replays**: MettaScope replay viewing
- **Experiment Log**: Structured sections for documenting iterations
- **Scratch Space**: Area for quick experiments

## Available Utilities

The notebooks have access to these key functions:

### Training
- `launch_training()` - Launch a single training run
- `launch_multiple_training_runs()` - Launch multiple runs with seed variation

### Monitoring
- `monitor_training_statuses()` - Check status of runs with live metrics
- `find_training_jobs()` - Find runs by author, tags, state, etc.

### Analysis
- `fetch_metrics()` - Pull metrics from wandb
- `plot_sps()` - Plot steps per second
- `create_run_summary_table()` - Generate summary statistics
- `get_run_config()` - Fetch run configuration
- `get_training_logs()` - Access stdout/stderr logs

### Replays
- `show_replay()` - Display MettaScope replay viewer
- `get_available_replays()` - List available replay steps

## Workflow

1. Create a new notebook with a descriptive name
2. Open in Jupyter/VS Code
3. Use the pre-built sections to:
   - Launch experiments
   - Monitor progress
   - Analyze results
   - Document findings
4. Iterate quickly with the provided tools

## Tips

- Keep experiment names descriptive and include dates
- Use the experiment tracking dict to organize multiple runs
- Document hypotheses and results in the Experiment Log section
- The notebook auto-saves your work as you go