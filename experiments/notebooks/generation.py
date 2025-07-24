"""Notebook generation utilities for experiments."""

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

from experiments.types import TrainingJob


# Available sections
AVAILABLE_SECTIONS = {
    "setup": "Setup and imports",
    "state": "State management for tracking runs",
    "launch": "Launch training runs", 
    "monitor": "Monitor training status",
    "metrics": "Fetch and analyze metrics",
    "visualize": "Visualizations and plots",
    "replays": "View MettaScope replays",
    "log": "Experiment log for documentation",
    "scratch": "Scratch space for experiments"
}

# Default sections if none specified
DEFAULT_SECTIONS = ["setup", "state", "launch", "monitor", "metrics", "visualize", "replays", "log", "scratch"]


def generate_notebook(
    name: str,
    description: str = "",
    sections: Optional[List[str]] = None,
    wandb_run_names: Optional[List[str]] = None,
    skypilot_job_ids: Optional[List[str]] = None,
    additional_metadata: Optional[Dict[str, Any]] = None,
    output_dir: str = "experiments/log",
) -> str:
    """Generate a research/experiment notebook.
    
    Args:
        name: Name for the notebook (will be used in filename)
        description: Optional description of the notebook purpose
        sections: List of sections to include (None = all sections)
        wandb_run_names: Optional pre-filled wandb run names (for experiments)
        skypilot_job_ids: Optional pre-filled sky job IDs (for experiments)
        additional_metadata: Optional metadata to include
        output_dir: Directory to save the notebook
        
    Returns:
        Path to the generated notebook
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.ipynb"
    filepath = os.path.join(output_dir, filename)
    
    # Use default sections if none specified
    if sections is None:
        sections = DEFAULT_SECTIONS
    
    # Create notebook structure
    notebook = {
        "cells": _create_notebook_cells(
            name=name,
            description=description,
            sections=sections,
            wandb_run_names=wandb_run_names,
            skypilot_job_ids=skypilot_job_ids,
            additional_metadata=additional_metadata
        ),
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
            "celltoolbar": "Tags",
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    
    # Write notebook
    with open(filepath, "w") as f:
        json.dump(notebook, f, indent=2)
    
    print(f"Generated notebook: {filepath}")
    if sections != DEFAULT_SECTIONS:
        print(f"Included sections: {', '.join(sections)}")
    
    return filepath


def generate_notebook_from_template(
    experiment_name: str,
    run_names: List[str],
    sky_job_ids: Optional[List[str]] = None,
    additional_metadata: Optional[Dict[str, Any]] = None,
    output_dir: str = "experiments/log",
) -> str:
    """Generate a notebook for analyzing experiment runs.
    
    This is a convenience wrapper for experiments that provides backwards compatibility.
    
    Args:
        experiment_name: Name of the experiment
        run_names: List of wandb run names
        sky_job_ids: Optional list of corresponding sky job IDs (skypilot_job_ids)
        additional_metadata: Optional additional metadata to include
        output_dir: Directory to save the notebook (default: experiments/log)

    Returns:
        Path to the generated notebook
    """
    return generate_notebook(
        name=experiment_name,
        description=f"Analysis notebook for {experiment_name} experiment",
        wandb_run_names=run_names,
        skypilot_job_ids=sky_job_ids,
        additional_metadata=additional_metadata,
        output_dir=output_dir
    )


def _create_notebook_cells(
    name: str,
    description: str,
    sections: List[str],
    wandb_run_names: Optional[List[str]] = None,
    skypilot_job_ids: Optional[List[str]] = None,
    additional_metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Create cells for a notebook based on selected sections."""
    cells = []
    
    # Title and description (always included)
    title = f"# {name.replace('_', ' ').title()}"
    if description:
        title += f"\n\n{description}"
    title += f"\n\nCreated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    cells.append(_create_markdown_cell(title))
    
    # If we have pre-filled IDs, add a summary cell
    if wandb_run_names:
        summary = f"""### Experiment Summary

**Experiment**: {name}  
**Runs**: {len(wandb_run_names)} training runs  
**Created**: {additional_metadata.get('created_at', 'Unknown') if additional_metadata else 'Unknown'}  
**User**: {additional_metadata.get('user', 'Unknown') if additional_metadata else 'Unknown'}

This notebook was auto-generated from the experiment run. The wandb run IDs and sky job IDs have been pre-loaded."""
        cells.append(_create_markdown_cell(summary))
    
    # Always include setup and state initialization first (marked as auto-run)
    cells.extend(_get_setup_section())
    cells.extend(_get_state_section(wandb_run_names, skypilot_job_ids, additional_metadata, name))
    
    # Generate cells for other requested sections
    section_generators = {
        "launch": _get_launch_section,
        "monitor": _get_monitor_section,
        "metrics": _get_metrics_section,
        "visualize": _get_visualize_section,
        "replays": _get_replays_section,
        "log": _get_log_section,
        "scratch": _get_scratch_section
    }
    
    for section in sections:
        # Skip setup and state since we already added them
        if section in section_generators:
            cells.extend(section_generators[section]())
    
    return cells


def _create_markdown_cell(content: str) -> Dict[str, Any]:
    """Create a markdown cell."""
    return {"cell_type": "markdown", "metadata": {}, "source": content}


def _create_code_cell(content: str) -> Dict[str, Any]:
    """Create a code cell."""
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": content}


def _get_setup_section() -> List[Dict[str, Any]]:
    """Generate setup section cells."""
    # Split into two cells - one for basic setup that should auto-run, one for imports
    cells = []
    
    # Auto-run cell with notebook configuration
    auto_run_cell = _create_code_cell("""# Notebook configuration (auto-run)
%load_ext autoreload
%autoreload 2
%matplotlib inline

import matplotlib.pyplot as plt
plt.style.use("default")

print("✓ Notebook configured")""")
    auto_run_cell["metadata"] = {"tags": ["auto-run"]}
    cells.append(auto_run_cell)
    
    # Regular imports cell
    cells.append(_create_code_cell("""# Standard imports
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# Metta experiment utilities  
from experiments.wandb_utils import find_training_jobs, get_run_config, get_training_logs
from experiments.monitoring import get_training_status
from experiments.notebooks.monitoring import monitor_training_statuses
from experiments.notebooks.replays import show_replay, get_available_replays
from experiments.notebooks.training import launch_training, launch_multiple_training_runs
from experiments.notebooks.analysis import fetch_metrics, plot_sps, create_run_summary_table

print(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")"""))
    
    return cells


def _get_state_section(
    wandb_run_names: Optional[List[str]] = None,
    skypilot_job_ids: Optional[List[str]] = None,
    additional_metadata: Optional[Dict[str, Any]] = None,
    experiment_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Generate state management section cells."""
    # No section header needed - this is just initialization
    cells = []
    
    # Initialize state with pre-filled data or empty
    init_code = f'''# Initialize run tracking (auto-run)
from experiments.notebooks.state import init_state, add_run, list_runs, kill_all_jobs

# Initialize with pre-loaded data
state = init_state(
    wandb_run_names={wandb_run_names},
    skypilot_job_ids={skypilot_job_ids or []},
    metadata={json.dumps(additional_metadata, indent=2) if additional_metadata else '{}'}
)

# Direct access to state data
wandb_run_names = state.wandb_run_names
skypilot_job_ids = state.skypilot_job_ids  
experiments = state.experiments

{f'print("✓ Loaded {len(wandb_run_names)} runs from {experiment_name or "experiment"}")' if wandb_run_names else 'print("✓ Run tracking initialized. Use add_run() to add runs.")'}
{'list_runs()' if wandb_run_names else ''}'''
    
    state_cell = _create_code_cell(init_code)
    state_cell["metadata"] = {"tags": ["auto-run"]}
    cells.append(state_cell)
    
    return cells


def _get_launch_section() -> List[Dict[str, Any]]:
    """Generate launch section cells."""
    cells = [_create_markdown_cell("## Launch Training")]
    
    # Uncommented, ready-to-use launch code
    launch_code = '''# Launch new training runs
# The result will contain both run_name and job_id

# Single run example:
run_name = f"{os.environ.get('USER')}.research.{datetime.now().strftime('%m%d_%H%M')}"
result = launch_training(
    run_name=run_name,
    curriculum="env/mettagrid/curriculum/arena/learning_progress",
    gpus=1,
    wandb_tags=["research", "experiment"],
    additional_args=[
        "trainer.optimizer.learning_rate=0.001",
        "trainer.optimizer.type=adam"
    ]
)

# Add to tracking
if result['success']:
    add_run(result['run_name'], result.get('job_id'))
    print(f"✓ Successfully launched {run_name}")
else:
    print(f"✗ Failed to launch {run_name}")'''
    
    cells.append(_create_code_cell(launch_code))
    
    # Additional example for multiple runs (still commented)
    multi_run_example = '''# Example: Multiple runs with seed variation
# base_name = f"{os.environ.get('USER')}.ablation.{datetime.now().strftime('%m%d_%H%M')}"
# results = launch_multiple_training_runs(
#     base_run_name=base_name,
#     curriculum="env/mettagrid/curriculum/arena/learning_progress",
#     num_runs=3,
#     vary_seeds=True,
#     gpus=1
# )
# 
# # Add all successful runs
# for result in results:
#     if result['success']:
#         add_run(result['run_name'], result.get('job_id'))'''
    
    cells.append(_create_code_cell(multi_run_example))
    
    return cells


def _get_monitor_section() -> List[Dict[str, Any]]:
    """Generate monitoring section cells."""
    return [
        _create_markdown_cell("## Monitor Training"),
        _create_code_cell('''# Monitor runs using the tracked IDs
# This will show both wandb and sky status if job IDs are available

# Monitor all tracked runs
# df = monitor_training_statuses(
#     wandb_run_ids, 
#     skypilot_job_ids=skypilot_job_ids,
#     show_metrics=["_step", "overview/reward"],
#     return_widget=True
# )

# Or find and monitor runs by criteria
# found_runs = find_training_jobs(
#     author=os.getenv("USER"),
#     wandb_tags=["research"],
#     state="running",
#     limit=10
# )
# df = monitor_training_statuses(found_runs)''')
    ]


def _get_metrics_section() -> List[Dict[str, Any]]:
    """Generate metrics section cells."""
    return [
        _create_markdown_cell("## Fetch & Analyze Metrics"),
        _create_code_cell('''# Fetch metrics for tracked runs
# metrics_dfs = fetch_metrics(wandb_run_names, samples=1000)

# Quick summary
# for run_id, df in metrics_dfs.items():
#     print(f"\\n{run_id}:")
#     print(f"  Steps: {df['_step'].max() if '_step' in df else 'N/A'}")
#     if 'overview/reward' in df:
#         print(f"  Final reward: {df['overview/reward'].iloc[-1]:.4f}")
#         print(f"  Max reward: {df['overview/reward'].max():.4f}")

# Create summary table
# summary_df = create_run_summary_table(wandb_run_names)
# print(summary_df)''')
    ]


def _get_visualize_section() -> List[Dict[str, Any]]:
    """Generate visualization section cells."""
    return [
        _create_markdown_cell("## Visualizations"),
        _create_code_cell('''# Plot metrics comparison
# Select metrics to plot
# plot_metrics = ["overview/reward", "losses/policy_loss", "losses/value_loss", "losses/entropy"]

# Create comparison plots
# fig = make_subplots(
#     rows=len(plot_metrics), 
#     cols=1,
#     subplot_titles=plot_metrics,
#     shared_xaxes=True,
#     vertical_spacing=0.05
# )

# # Add traces for each metric and run
# colors = ["blue", "red", "green", "orange", "purple"]
# for metric_idx, metric in enumerate(plot_metrics, 1):
#     for run_idx, (run_id, df) in enumerate(metrics_dfs.items()):
#         if metric in df.columns and '_step' in df.columns:
#             fig.add_trace(
#                 go.Scatter(
#                     x=df['_step'],
#                     y=df[metric],
#                     name=run_id.split('.')[-1],
#                     line=dict(color=colors[run_idx % len(colors)]),
#                     showlegend=(metric_idx == 1)
#                 ),
#                 row=metric_idx, col=1
#             )

# fig.update_layout(height=250 * len(plot_metrics), title="Metrics Comparison")
# fig.show()

# Plot SPS (Steps Per Second)
# fig_sps = plot_sps(wandb_run_names)
# fig_sps.show()''')
    ]


def _get_replays_section() -> List[Dict[str, Any]]:
    """Generate replays section cells."""
    return [
        _create_markdown_cell("## View Replays"),
        _create_code_cell('''# View replays for tracked runs
# if wandb_run_ids:
#     # Show last replay for first run
#     show_replay(wandb_run_names[0], step="last", width=1000, height=600)

# Get available replays
# if wandb_run_ids:
#     replays = get_available_replays(wandb_run_names[0])
#     for replay in replays[-5:]:  # Show last 5
#         print(f"{replay['label']} - Step {replay['step']}")''')
    ]


def _get_log_section() -> List[Dict[str, Any]]:
    """Generate experiment log section cells."""
    return [
        _create_markdown_cell("## Experiment Log\n\nDocument your findings and iterations here:"),
        _create_markdown_cell("""### Iteration 1

**Hypothesis:**

**Configuration:**
- Runs: 
- Key parameters:

**Results:**

**Next Steps:**""")
    ]


def _get_scratch_section() -> List[Dict[str, Any]]:
    """Generate scratch space section cells."""
    return [
        _create_markdown_cell("## Scratch Space"),
        _create_code_cell("# Quick experiments and one-off analysis\n")
    ]