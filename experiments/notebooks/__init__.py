"""Notebook utilities for experiments."""

# Import commonly used functions for easy access
from experiments.notebooks.analysis import (
    get_run_config,
    get_training_logs,
    plot_sps,
    create_run_summary_table,
)
from experiments.notebooks.generation import generate_notebook_from_template
from experiments.notebooks.metrics import fetch_metrics, find_training_jobs, get_run
from experiments.notebooks.monitoring import monitor_training_statuses
from experiments.notebooks.replays import show_replay, get_available_replays
from experiments.notebooks.training import launch_training, launch_multiple_training_runs

__all__ = [
    # Analysis
    "get_run_config",
    "get_training_logs",
    "plot_sps",
    "create_run_summary_table",
    # Generation
    "generate_notebook_from_template",
    # Metrics
    "fetch_metrics",
    "find_training_jobs",
    "get_run",
    # Monitoring
    "monitor_training_statuses",
    # Replays
    "show_replay",
    "get_available_replays",
    # Training
    "launch_training",
    "launch_multiple_training_runs",
]
