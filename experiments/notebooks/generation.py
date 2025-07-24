"""Utilities for generating analysis notebooks."""

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional


def generate_notebook_from_template(
    experiment_name: str,
    run_names: List[str],
    sky_job_ids: Optional[List[str]] = None,
    additional_metadata: Optional[Dict[str, Any]] = None,
    output_dir: str = "experiments/log",
) -> str:
    """Generate a Jupyter notebook for analyzing experiment runs.

    Args:
        experiment_name: Name of the experiment
        run_names: List of wandb run names
        sky_job_ids: Optional list of corresponding sky job IDs
        additional_metadata: Optional additional metadata to include
        output_dir: Directory to save the notebook (default: experiments/log)

    Returns:
        Path to the generated notebook
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    user = os.environ.get("USER", "unknown")
    filename = f"{experiment_name}_{timestamp}_{user}.ipynb"
    filepath = os.path.join(output_dir, filename)

    # Create notebook structure
    notebook = {
        "cells": [
            _create_markdown_cell(
                f"# {experiment_name.replace('_', ' ').title()} Analysis\n\nGenerated: {timestamp}\n\nUser: {user}"
            ),
            _create_markdown_cell("## Experiment Metadata"),
            _create_code_cell(_generate_metadata_code(run_names, sky_job_ids, additional_metadata)),
            _create_markdown_cell("## Setup"),
            _create_code_cell(_generate_setup_code()),
            _create_markdown_cell("## Run Configuration"),
            _create_code_cell(_generate_config_display_code()),
            _create_markdown_cell("## Training Logs"),
            _create_code_cell(_generate_logs_query_code()),
            _create_markdown_cell("## Steps Per Second (SPS)"),
            _create_code_cell(_generate_sps_plot_code()),
            _create_markdown_cell("## Run Summary"),
            _create_code_cell(_generate_summary_table_code()),
            _create_markdown_cell("## Additional Analysis\n\nAdd your custom analysis below:"),
            _create_code_cell("# Custom analysis code here\n"),
        ],
        "metadata": {
            "kernelspec": {"display_name": ".venv", "language": "python", "name": "python3"},
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.7",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    # Write notebook
    with open(filepath, "w") as f:
        json.dump(notebook, f, indent=2)

    print(f"Generated notebook: {filepath}")
    return filepath


def _create_markdown_cell(content: str) -> Dict[str, Any]:
    """Create a markdown cell."""
    return {"cell_type": "markdown", "metadata": {}, "source": content}


def _create_code_cell(content: str) -> Dict[str, Any]:
    """Create a code cell."""
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": content}


def _generate_metadata_code(
    run_names: List[str], sky_job_ids: Optional[List[str]], additional_metadata: Optional[Dict[str, Any]]
) -> str:
    """Generate code for displaying experiment metadata."""
    code_lines = [
        "# Experiment metadata",
        f"run_names = {run_names}",
    ]

    if sky_job_ids:
        code_lines.append(f"sky_job_ids = {sky_job_ids}")

    if additional_metadata:
        code_lines.append(f"metadata = {json.dumps(additional_metadata, indent=2)}")

    code_lines.extend(
        [
            "",
            'print(f"Analyzing {len(run_names)} training runs:")',
            "for i, run_name in enumerate(run_names):",
            '    print(f"  {i+1}. {run_name}")',
        ]
    )

    if sky_job_ids:
        code_lines.extend(
            [
                "    if i < len(sky_job_ids):",
                '        print(f"     Sky Job: {sky_job_ids[i]}")',
            ]
        )

    return "\n".join(code_lines)


def _generate_setup_code() -> str:
    """Generate setup code for the notebook."""
    return """# Setup
%load_ext autoreload
%autoreload 2

import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from experiments.notebooks.analysis import (
    get_run_config, get_training_logs, plot_sps, create_run_summary_table
)
from experiments.notebooks.metrics import fetch_metrics
from experiments.notebooks.monitoring import monitor_training_statuses
from experiments.notebooks.replays import show_replay

print("Setup complete!")"""


def _generate_config_display_code() -> str:
    """Generate code for displaying run configurations."""
    return """# Display configuration for each run
for run_name in run_names:
    print(f"\\nConfiguration for {run_name}:")
    print("=" * 50)
    config = get_run_config(run_name)
    if config:
        # Display key configuration parameters
        if 'trainer' in config:
            print(f"Curriculum: {config['trainer'].get('curriculum', 'N/A')}")
            print(f"Learning Rate: {config['trainer'].get('optimizer', {}).get('learning_rate', 'N/A')}")
            print(f"Optimizer: {config['trainer'].get('optimizer', {}).get('type', 'N/A')}")
        if 'hardware' in config:
            print(f"Hardware: {config.get('hardware', 'N/A')}")
    else:
        print("Failed to fetch configuration")"""


def _generate_logs_query_code() -> str:
    """Generate code for querying training logs."""
    return """# Query training logs
# Uncomment to fetch logs for a specific run
# run_to_check = run_names[0]
# logs = get_training_logs(run_to_check, log_type="stdout")
# print(f"Last 20 log lines for {run_to_check}:")
# for line in logs[-20:]:
#     print(line)"""


def _generate_sps_plot_code() -> str:
    """Generate code for plotting SPS."""
    return """# Plot Steps Per Second for all runs
fig = plot_sps(run_names, samples=1000)
fig.show()

# Monitor current status
if any(monitor_training_statuses(run_names)['state'] == 'running'):
    print("\\nNote: Some runs are still in progress. Refresh to see updated metrics.")"""


def _generate_summary_table_code() -> str:
    """Generate code for creating a summary table."""
    return """# Create summary table
summary_df = create_run_summary_table(run_names)
print("\\nRun Summary:")
print(summary_df.to_string(index=False))

# Optionally save to CSV
# summary_df.to_csv(f"{experiment_name}_summary.csv", index=False)"""
