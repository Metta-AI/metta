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
    "analysis": "Analysis and visualizations",
    "replays": "View MettaScope replays",
    "scratch": "Scratch space for experiments",
    "export": "Export notebook as HTML",
}

# Default sections if none specified
DEFAULT_SECTIONS = ["launch", "monitor", "analysis", "export"]


def generate_notebook(
    name: str,
    description: str = "",
    sections: Optional[List[str]] = None,
    wandb_run_names: Optional[List[str]] = None,
    skypilot_job_ids: Optional[List[str]] = None,
    additional_metadata: Optional[Dict[str, Any]] = None,
    output_dir: str = "experiments/scratch",
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
            additional_metadata=additional_metadata,
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
    output_dir: str = "experiments/scratch",
) -> str:
    """Generate a notebook for analyzing experiment runs.

    This is a convenience wrapper for experiments that provides backwards compatibility.

    Args:
        experiment_name: Name of the experiment
        run_names: List of wandb run names
        sky_job_ids: Optional list of corresponding sky job IDs (skypilot_job_ids)
        additional_metadata: Optional additional metadata to include
        output_dir: Directory to save the notebook (default: experiments/scratch)

    Returns:
        Path to the generated notebook
    """
    return generate_notebook(
        name=experiment_name,
        description=f"Analysis notebook for {experiment_name} experiment",
        wandb_run_names=run_names,
        skypilot_job_ids=sky_job_ids,
        additional_metadata=additional_metadata,
        output_dir=output_dir,
    )


def _create_notebook_cells(
    name: str,
    description: str,
    sections: List[str],
    wandb_run_names: Optional[List[str]] = None,
    skypilot_job_ids: Optional[List[str]] = None,
    additional_metadata: Optional[Dict[str, Any]] = None,
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
**Created**: {additional_metadata.get("created_at", datetime.now().strftime('%Y-%m-%d %H:%M'))}
**User**: {additional_metadata.get("user", "Unknown")}"""
        
        if skypilot_job_ids:
            summary += "\n\n**Tracked Jobs:**"
            for i, (job_id, run_name) in enumerate(zip(skypilot_job_ids, wandb_run_names)):
                summary += f"\n- Job {job_id} → {run_name}"
                    
        cells.append(_create_markdown_cell(summary))

    # Generate the notebook filename we'll use
    notebook_filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ipynb"

    # Create a single Setup section with all initialization
    cells.append(_create_markdown_cell("## Setup"))

    # Combine setup and state initialization into one section
    setup_cells = _get_setup_section()
    state_cells = _get_state_section(wandb_run_names, skypilot_job_ids, additional_metadata, name)

    # Mark all setup cells to be in the setup section
    for cell in setup_cells + state_cells:
        if cell["cell_type"] == "code":
            cell["metadata"]["tags"] = cell["metadata"].get("tags", []) + ["setup"]

    cells.extend(setup_cells)
    cells.extend(state_cells)

    # Generate cells for other requested sections
    has_existing_jobs = bool(wandb_run_names)  # True if we have pre-filled jobs
    section_generators = {
        "launch": lambda: _get_launch_section(has_existing_jobs=has_existing_jobs),
        "monitor": _get_monitor_section,
        "analysis": _get_analysis_section,
        "replays": _get_replays_section,
        "scratch": _get_scratch_section,
        "export": lambda: _get_export_section(notebook_filename),
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
    # Single comprehensive setup cell
    setup_cell = _create_code_cell("""# Initialize notebook
%load_ext autoreload
%autoreload 2

import os
from datetime import datetime
from experiments.notebooks.monitoring import job_status
from experiments.notebooks.training import launch_training
from experiments.notebooks.analysis import plot_sps

print("✓ Notebook initialized")""")

    return [setup_cell]


def _get_state_section(
    wandb_run_names: Optional[List[str]] = None,
    skypilot_job_ids: Optional[List[str]] = None,
    additional_metadata: Optional[Dict[str, Any]] = None,
    experiment_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Generate state management section cells."""
    # Initialize state with pre-filled data or empty
    init_code = f"""# Initialize run tracking
from experiments.notebooks.state import init_state, add_run, list_runs, kill_all_jobs
from experiments.types import TrainingJob

state = init_state(
    wandb_run_names={wandb_run_names},
    skypilot_job_ids={skypilot_job_ids or []},
    metadata={json.dumps(additional_metadata, indent=2) if additional_metadata else "{}"}
)

wandb_run_names = state.wandb_run_names
skypilot_job_ids = state.skypilot_job_ids
experiments = state.experiments

{f'print("✓ Ready. Tracking {len(wandb_run_names)} runs")' if wandb_run_names else 'print("✓ Ready")'}"""

    return [_create_code_cell(init_code)]


def _get_launch_section(has_existing_jobs: bool = False) -> List[Dict[str, Any]]:
    """Generate launch section cells."""
    if has_existing_jobs:
        # When loading existing jobs, show how to launch additional runs
        cells = [_create_markdown_cell("## Relaunch Training\n\n*Note: Jobs have been preloaded from command line. Use this section to launch additional training runs.*")]
        launch_code = """# Launch additional training runs to compare with preloaded jobs

# Example: Launch with different hyperparameters
run_name = f"{os.environ.get('USER')}.research.{datetime.now().strftime('%m%d_%H%M')}"
result = launch_training(
    run_name=run_name,
    curriculum="env/mettagrid/curriculum/arena/learning_progress",
    gpus=1,
    skip_git_check=False,  # Set to True to allow uncommitted changes
    wandb_tags=["research", "experiment", "comparison"],
    # no_spot=True is the default (more reliable for development)
)

# Add to tracking
if result['success']:
    add_run(result['run_name'], result.get('job_id'))
    print(f"✓ Successfully launched {run_name}")
else:
    print(f"✗ Failed to launch {run_name}")"""
    else:
        # Normal launch section for new notebooks
        cells = [_create_markdown_cell("## Launch Training")]
        launch_code = """# Launch new training runs
# The result will contain both run_name and job_id

# Single run example:
run_name = f"{os.environ.get('USER')}.research.{datetime.now().strftime('%m%d_%H%M')}"
result = launch_training(
    run_name=run_name,
    curriculum="env/mettagrid/curriculum/arena/learning_progress",
    gpus=1,
    skip_git_check=False,  # Set to True to allow uncommitted changes
    wandb_tags=["research", "experiment"]
    # no_spot=True is the default (more reliable for development)
)

# Add to tracking
if result['success']:
    add_run(result['run_name'], result.get('job_id'))
    print(f"✓ Successfully launched {run_name}")
else:
    print(f"✗ Failed to launch {run_name}")"""

    cells.append(_create_code_cell(launch_code))
    return cells


def _get_monitor_section() -> List[Dict[str, Any]]:
    """Generate monitoring section cells."""
    return [
        _create_markdown_cell("## Job Status"),
        _create_code_cell("""# Display status of all tracked runs
job_status(state.training_jobs)"""),
    ]




def _get_analysis_section() -> List[Dict[str, Any]]:
    """Generate analysis section cells."""
    return [
        _create_markdown_cell("## Analysis"),
        _create_code_cell("""# Plot SPS (Steps Per Second) to monitor training performance
fig_sps = plot_sps(wandb_run_names)
fig_sps.show()"""),
    ]


def _get_replays_section() -> List[Dict[str, Any]]:
    """Generate replays section cells."""
    return [
        _create_markdown_cell("## View Replays"),
        _create_code_cell("""# Show last replay for first run
if wandb_run_names:
    show_replay(wandb_run_names[0], step="last", width=1000, height=600)
else:
    print("No runs tracked yet. Launch some runs first!")"""),
        _create_code_cell("""# Get available replays for first run
if wandb_run_names:
    replays = get_available_replays(wandb_run_names[0])
    print(f"Available replays for {wandb_run_names[0]}:")
    for replay in replays[-10:]:  # Show last 10
        print(f"  {replay['label']} - Step {replay['step']}")"""),
    ]




def _get_scratch_section() -> List[Dict[str, Any]]:
    """Generate scratch space section cells."""
    return [_create_markdown_cell("## Scratch Space"), _create_code_cell("# Quick experiments and one-off analysis\n")]


def _get_export_section(notebook_filename: str) -> List[Dict[str, Any]]:
    """Generate export section cells."""
    html_export_code = f'''# Export this notebook as HTML to experiments/log/
from experiments.notebooks.export import export_to_html

notebook_name = "{notebook_filename}"
html_path = export_to_html(notebook_name)
print(f"Exported HTML to: {{html_path}}")'''

    clean_export_code = f'''# Export a clean (no outputs) version of this notebook
import subprocess
import shutil
from pathlib import Path

notebook_name = "{notebook_filename}"
clean_name = notebook_name.replace('.ipynb', '_clean.ipynb')

# Create a copy
shutil.copy(notebook_name, clean_name)

# Strip outputs using nbstripout
try:
    subprocess.run(['nbstripout', clean_name], check=True)
    print(f"✓ Created clean notebook: {{clean_name}}")
except subprocess.CalledProcessError:
    print("❌ Failed to run nbstripout. Install with: pip install nbstripout")
except FileNotFoundError:
    print("❌ nbstripout not found. Install with: pip install nbstripout")'''

    return [
        _create_markdown_cell("## Export Results"),
        _create_code_cell(html_export_code),
        _create_code_cell(clean_export_code)
    ]
