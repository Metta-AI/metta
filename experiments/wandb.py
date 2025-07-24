"""Weights & Biases utilities for experiments."""

from itertools import islice
from typing import List, Dict, Any, Optional
from datetime import datetime

import pandas as pd
import wandb
from wandb.apis.public.runs import Run


def get_run(run_name: str, entity: str = "metta-research", project: str = "metta") -> Run | None:
    """Get a wandb run by name.
    
    Args:
        run_name: Name of the wandb run
        entity: Wandb entity
        project: Wandb project
        
    Returns:
        Run object or None if not found
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
    """Find training jobs matching criteria.
    
    Args:
        wandb_tags: Filter by tags
        author: Filter by username
        state: Filter by state (running, finished, failed, etc.)
        created_after: Filter by creation date
        created_before: Filter by creation date
        entity: Wandb entity
        project: Wandb project
        order_by: Sort order
        limit: Maximum number of results
        
    Returns:
        List of run names
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


def fetch_metrics_data(
    run_names: list[str], 
    samples: int = 1000,
    entity: str = "metta-research",
    project: str = "metta"
) -> dict[str, pd.DataFrame]:
    """Fetch metrics data for multiple runs.
    
    Args:
        run_names: List of run names to fetch
        samples: Number of samples to fetch per run
        entity: Wandb entity
        project: Wandb project
        
    Returns:
        Dictionary mapping run names to dataframes
    """
    metrics_dfs = {}

    for run_name in run_names:
        run = get_run(run_name, entity, project)
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


def get_run_statuses(
    run_names: list[str],
    show_metrics: list[str] | None = None,
    entity: str = "metta-research", 
    project: str = "metta"
) -> pd.DataFrame:
    """Get status information for multiple runs.
    
    Args:
        run_names: List of run names
        show_metrics: Metrics to include in status
        entity: Wandb entity
        project: Wandb project
        
    Returns:
        DataFrame with run status information
    """
    if show_metrics is None:
        show_metrics = ["_step", "overview/reward"]

    runs = wandb.Api().runs(f"{entity}/{project}", filters={"name": {"$in": run_names}})

    # Collect data for each run
    data = []
    for run_name in run_names:
        run = next((r for r in runs if r.name == run_name), None)
        row = {
            "run_name": run_name,
            "state": "NOT FOUND",
            "created": None,
            "url": None,
        }

        if run:
            row.update({
                "run_name": run_name,
                "state": run.state,
                "created": datetime.fromisoformat(run.created_at).strftime("%Y-%m-%d %H:%M"),
            })
            
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

    return pd.DataFrame(data)


def get_run_config(run_name: str, entity: str = "metta-research", project: str = "metta") -> Dict[str, Any]:
    """Fetch full configuration from a wandb run.
    
    Args:
        run_name: Name of the wandb run
        entity: Wandb entity
        project: Wandb project
        
    Returns:
        Run configuration dictionary
    """
    try:
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_name}")
        return run.config
    except Exception as e:
        print(f"Error fetching config for {run_name}: {str(e)}")
        return {}


def get_training_logs(
    run_name: str, 
    log_type: str = "stdout", 
    entity: str = "metta-research", 
    project: str = "metta"
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