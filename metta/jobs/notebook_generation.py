"""Notebook generation utilities for training experiments.

Generates Jupyter notebooks with WandB metric visualizations for training jobs.
"""

import json
from pathlib import Path

from metta.common.util.constants import METTA_WANDB_ENTITY, METTA_WANDB_PROJECT
from metta.jobs.job_state import JobState


def generate_experiment_notebook(
    notebook_path: Path,
    group_name: str,
    job_states: list[JobState],
    entity: str = METTA_WANDB_ENTITY,
    project: str = METTA_WANDB_PROJECT,
) -> None:
    """Generate a Jupyter notebook with reward and SPS graphs for training jobs.

    Creates a notebook that fetches metrics from WandB and generates visualizations
    for all jobs in an experiment.

    Args:
        notebook_path: Path where the notebook will be saved
        group_name: Name of the experiment/job group
        job_states: List of job states from the experiment
        entity: WandB entity (default from constants)
        project: WandB project (default from constants)
    """
    cells = []

    # Title cell
    run_names = [state.name for state in job_states]
    cells.append(_create_title_cell(group_name, run_names))

    # Setup cell with imports and WandB config
    cells.append(_create_setup_cell(run_names, entity, project))

    # WandB URLs cell (if available)
    wandb_urls = [(state.name, state.wandb_url) for state in job_states if state.wandb_url]
    if wandb_urls:
        cells.append(_create_wandb_urls_cell(wandb_urls))

    # Fetch metrics cell
    cells.append(_create_fetch_metrics_cell())

    # Graph generation cell
    cells.append(_create_graphs_cell())

    # Summary stats cell
    cells.append(_create_summary_cell())

    # Create notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    # Ensure parent directory exists
    notebook_path.parent.mkdir(parents=True, exist_ok=True)

    # Write notebook
    with open(notebook_path, "w") as f:
        json.dump(notebook, f, indent=2)


def _create_title_cell(group_name: str, run_names: list[str]) -> dict:
    """Create markdown title cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# Experiment: {group_name}\n",
            "\n",
            f"**Runs**: {', '.join(run_names)}\n",
            "\n",
            "This notebook shows reward and SPS metrics for the experiment runs.\n",
        ],
    }


def _create_setup_cell(run_names: list[str], entity: str, project: str) -> dict:
    """Create setup cell with imports and configuration."""
    return {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Setup: Import libraries and configure WandB\n",
            "from experiments.notebooks.utils.metrics import fetch_metrics\n",
            "import matplotlib.pyplot as plt\n",
            "import pandas as pd\n",
            "\n",
            "# Experiment configuration\n",
            f"ENTITY = '{entity}'\n",
            f"PROJECT = '{project}'\n",
            f"RUN_NAMES = {run_names}\n",
            "\n",
            "print(f'Configured for {len(RUN_NAMES)} runs')\n",
        ],
        "execution_count": None,
        "outputs": [],
    }


def _create_wandb_urls_cell(wandb_urls: list[tuple[str, str]]) -> dict:
    """Create cell with WandB run URLs."""
    url_lines = [f"- [{name}]({url})\n" for name, url in wandb_urls]
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## WandB Run URLs\n", "\n"] + url_lines,
    }


def _create_fetch_metrics_cell() -> dict:
    """Create cell to fetch metrics from WandB."""
    return {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Fetch metrics from WandB\n",
            "# Uses fetch_metrics from experiments.notebooks.utils.metrics\n",
            "# This fetches sampled data (1000 points by default) for faster loading\n",
            "\n",
            "metrics_dfs = fetch_metrics(\n",
            "    run_names=RUN_NAMES,\n",
            "    samples=1000,  # Sampled data for faster loading\n",
            "    keys=['overview/reward', 'overview/sps', '_step'],  # Specific metrics\n",
            ")\n",
            "\n",
            "print(f'Fetched metrics for {len(metrics_dfs)} runs')\n",
        ],
        "execution_count": None,
        "outputs": [],
    }


def _create_graphs_cell() -> dict:
    """Create cell to generate reward and SPS graphs."""
    return {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Generate reward and SPS graphs\n",
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))\n",
            "\n",
            "for run_name, df in metrics_dfs.items():\n",
            "    if len(df) == 0:\n",
            "        print(f'Warning: No data for {run_name}')\n",
            "        continue\n",
            "    \n",
            "    # Reward plot\n",
            "    if 'overview/reward' in df.columns and '_step' in df.columns:\n",
            "        ax1.plot(df['_step'], df['overview/reward'], label=run_name, alpha=0.7)\n",
            "    \n",
            "    # SPS plot\n",
            "    if 'overview/sps' in df.columns and '_step' in df.columns:\n",
            "        ax2.plot(df['_step'], df['overview/sps'], label=run_name, alpha=0.7)\n",
            "\n",
            "# Configure reward plot\n",
            "ax1.set_xlabel('Training Step')\n",
            "ax1.set_ylabel('Reward')\n",
            "ax1.set_title('Agent Reward Over Time')\n",
            "ax1.legend()\n",
            "ax1.grid(True, alpha=0.3)\n",
            "\n",
            "# Configure SPS plot\n",
            "ax2.set_xlabel('Training Step')\n",
            "ax2.set_ylabel('Steps Per Second')\n",
            "ax2.set_title('Training Throughput (SPS)')\n",
            "ax2.legend()\n",
            "ax2.grid(True, alpha=0.3)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
        ],
        "execution_count": None,
        "outputs": [],
    }


def _create_summary_cell() -> dict:
    """Create cell with summary statistics."""
    return {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Summary statistics\n",
            "summary_data = []\n",
            "\n",
            "for run_name, df in metrics_dfs.items():\n",
            "    if len(df) == 0:\n",
            "        continue\n",
            "    \n",
            "    summary = {'run': run_name}\n",
            "    \n",
            "    if 'overview/reward' in df.columns:\n",
            "        rewards = df['overview/reward'].dropna()\n",
            "        if len(rewards) > 0:\n",
            "            summary['final_reward'] = rewards.iloc[-1]\n",
            "            summary['max_reward'] = rewards.max()\n",
            "            summary['mean_reward'] = rewards.mean()\n",
            "    \n",
            "    if 'overview/sps' in df.columns:\n",
            "        sps = df['overview/sps'].dropna()\n",
            "        if len(sps) > 0:\n",
            "            summary['avg_sps'] = sps.mean()\n",
            "            summary['max_sps'] = sps.max()\n",
            "    \n",
            "    if '_step' in df.columns:\n",
            "        steps = df['_step'].dropna()\n",
            "        if len(steps) > 0:\n",
            "            summary['final_step'] = int(steps.iloc[-1])\n",
            "    \n",
            "    summary_data.append(summary)\n",
            "\n",
            "# Display as DataFrame\n",
            "if summary_data:\n",
            "    summary_df = pd.DataFrame(summary_data)\n",
            "    display(summary_df)\n",
            "else:\n",
            "    print('No summary data available')\n",
        ],
        "execution_count": None,
        "outputs": [],
    }
