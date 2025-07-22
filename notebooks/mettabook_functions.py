import json
import os
import re
import subprocess
from datetime import datetime
from itertools import islice
from typing import Any

import ipywidgets as widgets
import pandas as pd
import wandb
import yaml
from IPython.display import IFrame, display
from wandb.apis.public.runs import Run

from metta.common.util.collections import remove_none_values
from metta.common.util.fs import get_repo_root


def get_run(run_name: str, entity: str = "metta-research", project: str = "metta") -> Run | None:
    """
    Get a W&B run object by name.

    Args:
        run_name: W&B run name
        api: Optional W&B API instance (will create one if not provided)

    Returns:
        W&B Run object or None if error
    """
    try:
        api = wandb.Api()
    except Exception as e:
        print(f"Error connecting to W&B: {str(e)}")
        print("Make sure you are connected to W&B: `metta status`")
        return None

    try:
        return api.run(f"{entity}/{project}/{run_name}")
    except Exception as e:
        print(f"Error getting run {run_name}: {str(e)}")
        return None


def _load_available_environments():
    config_path = os.path.join(get_repo_root(), "configs", "sim", "all.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    environments = []
    if "simulations" in config:
        for sim_config in config["simulations"].values():
            if "env" in sim_config:
                env_path = sim_config["env"]
                environments.append(env_path)
    return environments


def launch_training(
    run_name: str,
    num_gpus: int | None = None,
    num_cpus: int | None = None,
    no_spot: bool | None = None,
    curriculum: str | None = None,
    git_ref: str | None = None,
    skip_git_check: bool | None = None,
    additional_args: list[str] | None = None,
    dry_run: bool | None = None,
    wandb_tags: list[str] | None = None,
) -> dict:
    """
    Launch a training job on SkyPilot.

    Args:
        run_name: Name for the training run
        num_gpus: Number of GPUs to request
        num_cpus: Number of CPUs to request
        use_spot: Whether to use spot instances
        curriculum: Curriculum to use (defaults to first available)
        additional_args: Additional training arguments

    Returns:
        dict with keys: 'job_id', 'job_name', 'success', 'command', 'output'
    """
    if curriculum and curriculum not in _load_available_environments():
        raise ValueError(f"Curriculum {curriculum} not found. Available environments: {_load_available_environments()}")

    cmd_args = remove_none_values(
        {
            "gpu": num_gpus,
            "cpu": num_cpus,
            "no_spot": no_spot,
            "git_ref": git_ref,
            "skip_git_check": skip_git_check,
            "dry_run": dry_run,
        }
    )

    cmd = [
        "./devops/skypilot/launch.py",
        "train",
        f"run={run_name}",
        *[f"--{k}={v}" for k, v in cmd_args.items()],
    ]

    if curriculum:
        cmd.append(f"trainer.curriculum={curriculum}")
    if wandb_tags:
        cmd.append(f"+wandb.tags={json.dumps(wandb_tags)}")

    if additional_args:
        cmd.extend(additional_args)

    print(f"Launching training job: {run_name}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 50)

    result = {
        "job_id": None,
        "job_name": run_name,
        "success": False,
        "command": " ".join(cmd),
        "output": [],
    }

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=get_repo_root(),
        )

        for line in process.stdout or []:
            result["output"].append(line.strip())
            print(line, end="")
            if "Job ID:" in line or "sky-" in line:
                parts = line.split()
                for part in parts:
                    if part.startswith("sky-") and "-" in part[4:]:
                        result["job_id"] = part

        process.wait()
        result["success"] = process.returncode == 0

        if result["success"]:
            print("\n✓ Job launched successfully!")
            if result["job_id"]:
                print(f"Job ID: {result['job_id']}")
        else:
            print(f"\n✗ Launch failed with return code: {process.returncode}")

    except Exception as e:
        print(f"\n✗ Error launching job: {str(e)}")
        result["output"].append(f"Error: {str(e)}")

    return result


def find_training_jobs(
    wandb_tags: list[str] | None = None,
    author: str | None = None,
    state: str | None = None,
    created_after: str | None = None,
    created_before: str | None = None,
    entity: str = "metta-research",
    project: str = "metta",
    order_by: str = "-created_at",
    limit: int = 50,
) -> list[str]:
    """
    Search for training jobs based on various criteria.

    Args:
        wandb_tags: List of tags to filter by (runs must have ALL tags)
        author: Filter by run author/username
        state: Filter by run state ('running', 'finished', 'failed', 'crashed')
        created_after: ISO format date string (e.g., '2024-07-01')
        created_before: ISO format date string
        entity: W&B entity name
        project: W&B project name
        limit: Maximum number of runs to return

    Returns:
        List of run names matching the criteria
    """
    filters = {}
    if state:
        filters["state"] = state
    if author:
        filters["username"] = author
    if created_after:
        filters["created_at"] = {"$gte": created_after}

    if created_before:
        if "created_at" in filters:
            filters["created_at"]["$lte"] = created_before
        else:
            filters["created_at"] = {"$lte": created_before}
    if wandb_tags:
        filters["tags"] = {"$in": wandb_tags}
    runs = islice(wandb.Api().runs(f"{entity}/{project}", filters=filters, order=order_by), limit)

    return [run.name for run in runs]


def monitor_training_statuses(
    run_names: list[str],
    show_metrics: list[str] | None = None,
    entity: str = "metta-research",
    project: str = "metta",
) -> pd.DataFrame:
    """
    Monitor the status of multiple training runs in a table format.

    Args:
        run_names: List of W&B run names to monitor
        show_metrics: List of metric names to display (e.g., ['overview/reward', '_step'])
                     If None, shows default metrics
        entity: W&B entity name
        project: W&B project name
        fetch_data: If False, only shows run names and URLs without fetching run data

    Returns:
        DataFrame with run statuses and metrics
    """
    if show_metrics is None:
        show_metrics = ["_step", "overview/reward"]

    runs = wandb.Api().runs(f"{entity}/{project}", filters={"name": {"$in": run_names}})

    # Collect data for each run
    data = []
    for run_name in run_names:
        run = next((r for r in runs if r.name == run_name), None)
        row = {
            "name": run_name,
            "state": "NOT FOUND",
            "created": None,
            "url": None,
        }
        if run:
            row.update(
                {
                    "name": run_name,
                    "state": run.state,
                    "created": datetime.fromisoformat(run.created_at).strftime("%Y-%m-%d %H:%M"),
                }
            )
            if run.summary:
                for metric in show_metrics:
                    if metric in run.summary:
                        value = run.summary[metric]
                        if isinstance(value, float):
                            row[metric] = f"{value:.4f}"
                        else:
                            row[metric] = value
                    else:
                        row[metric] = "-"
            else:
                for metric in show_metrics:
                    row[metric] = "-"
            row["url"] = run.url
        data.append(row)

    df = pd.DataFrame(data)

    if not df.empty:
        _display_training_table_widget(df)

    return df


def _display_training_table_widget(df: pd.DataFrame) -> None:
    """Display training table as interactive widget in Jupyter."""

    # Create styled HTML table
    html_rows = []

    def wrap_with_component(component: str, value: Any, additional_style: str = "") -> str:
        return f"<{component} style='padding: 8px; text-align: right; {additional_style}'>{value}</{component}>"

    # Header
    header_html = (
        "<tr>" + "".join(wrap_with_component("th", h, "background-color: #f0f0f0;") for h in df.columns) + "</tr>"
    )

    # Rows with styling
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            value = row[col]

            # Special styling for different columns
            if col == "state":
                color = {
                    "running": "#28a745",
                    "finished": "#007bff",
                    "failed": "#dc3545",
                    "crashed": "#dc3545",
                    "NOT FOUND": "#ffc107",
                }.get(str(value), "#000")
                cell_html = wrap_with_component("td", value, f"color: {color}; font-weight: bold;")
            elif col == "url" and bool(value):
                cell_html = wrap_with_component(
                    "td",
                    f"<a href='{value}' target='_blank' style='text-decoration: none;'>wandb link</a>",
                )
            else:
                cell_html = wrap_with_component("td", value if value is not None else "-")
            cells.append(cell_html)

        html_rows.append("<tr style='border-bottom: 1px solid #ddd;'>" + "".join(cells) + "</tr>")

    # Create complete table HTML
    table_html = f"""
    <style>
        .training-table {{
            border-collapse: collapse;
            width: 100%;
            font-family: Arial, sans-serif;
            font-size: 14px;
        }}
        .training-table tr:hover {{
            background-color: #f5f5f5;
        }}
    </style>
    <table class='training-table'>
        <thead>{header_html}</thead>
        <tbody>{"".join(html_rows)}</tbody>
    </table>
    """

    display(widgets.HTML(table_html))


def fetch_metrics(run_names: list[str], samples: int = 1000) -> dict[str, pd.DataFrame]:
    """
    Fetch metrics from W&B for multiple runs.

    Args:
        run_names: List of W&B run names
        last_n_points: Number of most recent points to fetch
        sample_rate: Sample every Nth data point (1 = all data)

    Returns:
        Dictionary mapping run_name -> metrics DataFrame
    """
    metrics_dfs = {}

    for run_name in run_names:
        run = get_run(run_name)
        if run is None:
            continue

        print(f"Fetching metrics for {run_name}: {run.state}, {run.created_at}\n{run.url}...")

        try:
            metrics_df: pd.DataFrame = run.history(samples=samples, pandas=True)  # type: ignore
            metrics_dfs[run_name] = metrics_df
            print(f"  Fetched {len(metrics_df)} data points.")

            if len(metrics_df) > 0 and "overview/reward" in metrics_df.columns:
                print(
                    f"  Reward: mean={metrics_df['overview/reward'].mean():.4f}, "
                    f"max={metrics_df['overview/reward'].max():.4f}"
                )
            print(f"  Access with `metrics_dfs['{run_name}']`")
            print("")

        except Exception as e:
            print(f"  Error: {str(e)}")
    return metrics_dfs


def show_replay(run_name: str, step: str | int = "last", width: int = 1000, height: int = 600) -> None:
    """
    Display a replay viewer for a specific run and step.

    Args:
        run_name: W&B run name
        step: "last" for most recent, "first" for earliest, or specific step number
        width: IFrame width in pixels
        height: IFrame height in pixels
    """
    run = get_run(run_name)
    if run is None:
        return

    replay_urls = _fetch_replay_urls_for_run(run)

    if not replay_urls:
        print(f"No replays found for {run_name}")
        return

    # Select the requested replay
    if step == "last":
        selected = replay_urls[-1]
    elif step == "first":
        selected = replay_urls[0]
    else:
        # Find replay closest to requested step
        target_step = int(step)
        selected = min(replay_urls, key=lambda r: abs(r["step"] - target_step))
        if selected["step"] != target_step:
            print(f"Note: Requested step {target_step}, showing closest available step {selected['step']}")

    print(f"Loading MettaScope viewer for {run_name} at step {selected['step']:,}...")
    print(f"\nDirect link: {selected['url']}")
    display(IFrame(src=selected["url"], width=width, height=height))


def get_available_replays(run_name: str) -> list[dict]:
    """
    Get list of available replay steps for a run.

    Args:
        run_name: W&B run name

    Returns:
        List of dicts with keys: 'step', 'url', 'label'
    """
    run = get_run(run_name)
    if run is None:
        return []

    return _fetch_replay_urls_for_run(run)


def _fetch_replay_urls_for_run(run) -> list[dict]:
    """Fetch replay URLs for a single W&B run."""
    files = run.files()
    replay_urls = []

    # Filter for replay HTML files
    replay_files = [f for f in files if "media/html/replays/link_" in f.name and f.name.endswith(".html")]

    if not replay_files:
        return []

    # Sort by step number
    def get_step_from_filename(file):
        match = re.search(r"link_(\d+)_", file.name)
        return int(match.group(1)) if match else 0

    replay_files.sort(key=get_step_from_filename)

    # Process files (limit to avoid too many)
    max_files = min(20, len(replay_files))
    recent_files = replay_files[-max_files:]

    for file in recent_files:
        try:
            # Download and read the HTML file
            with file.download(replace=True, root="/tmp") as f:
                content = f.read()
            match = re.search(r'<a[^>]+href="([^"]+)"', content)
            if match:
                href = match.group(1)
                if href:
                    step = get_step_from_filename(file)
                    replay_urls.append({"step": step, "url": href, "filename": file.name, "label": f"Step {step:,}"})
        except Exception:
            pass

    return replay_urls
