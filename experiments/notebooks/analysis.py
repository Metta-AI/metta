"""Analysis and visualization utilities for experiment notebooks."""

from typing import List, Dict, Any, Optional
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import data fetching functions from experiments
from experiments.wandb import fetch_metrics_data, get_run_config, get_training_logs


def fetch_metrics(wandb_run_ids: list[str], samples: int = 1000) -> dict[str, pd.DataFrame]:
    """Fetch metrics for analysis (notebook-friendly wrapper).
    
    Args:
        wandb_run_ids: List of wandb run names
        samples: Number of samples to fetch
        
    Returns:
        Dictionary mapping run names to dataframes
    """
    return fetch_metrics_data(wandb_run_ids, samples)


def plot_sps(
    wandb_run_ids: List[str],
    samples: int = 1000,
    entity: str = "metta-research",
    project: str = "metta",
    title: str = "Steps Per Second",
    width: int = 1000,
    height: int = 600,
) -> go.Figure:
    """Plot steps per second for one or more runs.

    Args:
        wandb_run_ids: List of wandb run names to plot
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

    for idx, run_name in enumerate(wandb_run_ids):
        try:
            from experiments.wandb import get_run
            run = get_run(run_name, entity, project)
            if run is None:
                continue

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
    wandb_run_ids: List[str], metrics: Optional[List[str]] = None, entity: str = "metta-research", project: str = "metta"
) -> pd.DataFrame:
    """Create a summary table for multiple runs.

    Args:
        wandb_run_ids: List of wandb run names
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
    
    from experiments.wandb import get_run
    for run_name in wandb_run_ids:
        try:
            run = get_run(run_name, entity, project)
            if run is None:
                data.append({"run_name": run_name, "state": "NOT FOUND"})
                continue
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
