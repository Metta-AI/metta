"""Analysis utilities for experiment notebooks."""

from typing import List, Dict, Any, Optional
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import wandb
from wandb.apis.public.runs import Run


def get_run_config(run_name: str, entity: str = "metta-research", project: str = "metta") -> Dict[str, Any]:
    """Fetch full configuration from a wandb run."""
    try:
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_name}")
        return run.config
    except Exception as e:
        print(f"Error fetching config for {run_name}: {str(e)}")
        return {}


def get_training_logs(
    run_name: str, log_type: str = "stdout", entity: str = "metta-research", project: str = "metta"
) -> List[str]:
    """Fetch training logs (stdout/stderr) from a wandb run.

    Args:
        run_name: Name of the wandb run
        log_type: Either "stdout" or "stderr"
        entity: Wandb entity
        project: Wandb project

    Returns:
        List of log lines
    """
    try:
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_name}")

        # Get log files
        files = run.files()
        log_filename = f"output.log" if log_type == "stdout" else f"error.log"

        for file in files:
            if file.name.endswith(log_filename):
                # Download and read the log file
                with file.download(replace=True, root="/tmp") as f:
                    return f.read().decode("utf-8").splitlines()

        print(f"No {log_type} logs found for {run_name}")
        return []
    except Exception as e:
        print(f"Error fetching logs for {run_name}: {str(e)}")
        return []


def plot_sps(
    run_names: List[str],
    samples: int = 1000,
    entity: str = "metta-research",
    project: str = "metta",
    title: str = "Steps Per Second",
    width: int = 1000,
    height: int = 600,
) -> go.Figure:
    """Plot steps per second for one or more runs.

    Args:
        run_names: List of wandb run names to plot
        samples: Number of samples to fetch
        entity: Wandb entity
        project: Wandb project
        title: Plot title
        width: Plot width
        height: Plot height

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    # Color palette for different runs
    colors = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]

    for idx, run_name in enumerate(run_names):
        try:
            api = wandb.Api()
            run = api.run(f"{entity}/{project}/{run_name}")

            # Fetch history
            history_df = run.history(samples=samples, pandas=True)

            # Look for SPS metric - common variations
            sps_column = None
            for col in [
                "sps",
                "steps_per_second",
                "train/sps",
                "train/steps_per_second",
                "timing/sps",
                "timing/steps_per_second",
                "performance/sps",
            ]:
                if col in history_df.columns:
                    sps_column = col
                    break

            if sps_column and "_step" in history_df.columns:
                color = colors[idx % len(colors)]

                fig.add_trace(
                    go.Scatter(
                        x=history_df["_step"],
                        y=history_df[sps_column],
                        mode="lines",
                        name=run_name,
                        line=dict(color=color, width=2),
                    )
                )
            else:
                print(f"No SPS metric found for {run_name}")

        except Exception as e:
            print(f"Error plotting SPS for {run_name}: {str(e)}")

    fig.update_layout(
        title=title,
        xaxis_title="Steps",
        yaxis_title="Steps Per Second",
        width=width,
        height=height,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_run_summary_table(
    run_names: List[str], metrics: Optional[List[str]] = None, entity: str = "metta-research", project: str = "metta"
) -> pd.DataFrame:
    """Create a summary table for multiple runs.

    Args:
        run_names: List of wandb run names
        metrics: List of metrics to include (if None, includes common ones)
        entity: Wandb entity
        project: Wandb project

    Returns:
        DataFrame with run summaries
    """
    if metrics is None:
        metrics = [
            "overview/reward",
            "losses/policy_loss",
            "losses/value_loss",
            "losses/entropy",
            "losses/explained_variance",
        ]

    data = []
    api = wandb.Api()

    for run_name in run_names:
        try:
            run = api.run(f"{entity}/{project}/{run_name}")
            row = {
                "run_name": run_name,
                "state": run.state,
                "duration": run.summary.get("_runtime", 0) / 3600,  # hours
                "total_steps": run.summary.get("_step", 0),
            }

            # Add requested metrics
            for metric in metrics:
                if metric in run.summary:
                    value = run.summary[metric]
                    if isinstance(value, float):
                        row[metric] = round(value, 4)
                    else:
                        row[metric] = value
                else:
                    row[metric] = None

            data.append(row)

        except Exception as e:
            print(f"Error fetching summary for {run_name}: {str(e)}")
            data.append({"run_name": run_name, "state": "ERROR"})

    return pd.DataFrame(data)
