"""Notebook utilities for experiments."""

# Import commonly used functions for easy access
from experiments.notebooks.analysis import (
    fetch_metrics,
    plot_sps,
    create_run_summary_table,
)
from experiments.notebooks.generation import generate_notebook_from_template
from experiments.notebooks.monitoring import monitor_training_statuses
from experiments.notebooks.replays import show_replay, get_available_replays
from experiments.notebooks.training import launch_training, launch_multiple_training_runs

# Import from experiments package
from experiments.wandb_utils import find_training_jobs, get_run, get_run_config, get_training_logs

__all__ = [
    # Analysis
    "fetch_metrics",
    "plot_sps",
    "create_run_summary_table",
    # Generation
    "generate_notebook_from_template",
    # Monitoring
    "monitor_training_statuses",
    # Replays
    "show_replay",
    "get_available_replays",
    # Training
    "launch_training",
    "launch_multiple_training_runs",
    # From experiments.wandb
    "find_training_jobs",
    "get_run",
    "get_run_config",
    "get_training_logs",
]
