import json
import logging
from typing import Any, List

import wandb

from metta.common.util.numpy_helpers import clean_numpy_types

logger = logging.getLogger("sweep")


# 1 - Sweep utilities.
def get_sweep_runs(sweep_name: str, entity: str, project: str) -> List[Any]:
    """Get all runs from a sweep (group) sorted by score."""
    api = wandb.Api()

    # Get all runs from the group
    runs = api.runs(
        f"{entity}/{project}",
        filters={
            "group": sweep_name,
            "state": "finished",  # Only successful runs
        },
    )

    # Filter for runs with valid scores
    valid_runs = []
    for run in runs:
        score = run.summary.get("score", run.summary.get("protein.objective", 0))
        if score is not None and score > 0:  # Filter out failed runs
            valid_runs.append(run)

    # Sort by score (descending for reward metric)
    valid_runs.sort(key=lambda r: r.summary.get("score", r.summary.get("protein.objective", 0)), reverse=True)
    return valid_runs


# 2 - Protein Integration Utilities.
def fetch_protein_observations_from_wandb(
    wandb_entity: str,
    wandb_project: str,
    sweep_name: str,
    max_observations: int = 100,
) -> list[dict[str, Any]]:
    """
    Fetch latest protein observations from WandB sweep runs using groups.

    Args:
        wandb_entity: The WandB entity name.
        wandb_project: The WandB project name.
        sweep_name: The sweep name (used as group).
        max_observations: The maximum number of observations to fetch.

    Returns:
        List of observation dictionaries with format:
        {
            "suggestion": dict,      # The hyperparameters used
            "objective": float,      # The objective value achieved
            "cost": float,          # The cost (e.g., runtime in seconds)
            "is_failure": bool,     # Whether the run failed

        }
     ."""
    api = wandb.Api()
    wandb_path = f"{wandb_entity}/{wandb_project}"

    # Get runs from this sweep group
    runs = api.runs(
        path=wandb_path,
        filters={
            "group": sweep_name,  # Filter by group
            "state": {"$in": ["finished", "failed"]},  # Only completed runs
        },
        order="-created_at",  # Newest first
    )

    # Extract protein observations from runs
    observations = []
    for run in runs[:max_observations]:  # Limit to max_observations
        protein_obs = run.summary.get("protein_observation")
        if protein_obs is not None:
            observations.append(deep_clean(protein_obs))

    return observations


def record_protein_observation_to_wandb(
    wandb_run: Any,
    suggestion: dict[str, Any],
    objective: float,
    cost: float,
    is_failure: bool,
) -> None:
    """
    Record an observation to WandB.

    Args:
        wandb_run: The WandB run to record the observation to.
        suggestion: The suggestioån to record.
        objective: The objective value to optimize (higher is better for maximization).
        cost: The cost of this evaluation (e.g., time taken).
        is_failure: Whether the suggestion failed.
    """
    wandb_run.summary.update(
        {
            "protein_observation": {
                "suggestion": suggestion,
                "objective": objective,
                "cost": cost,
                "is_failure": is_failure,
            },
        }
    )


# 3 - Data Utilities.
def deep_clean(obj):
    """Recursively convert any object to JSON-serializable Python types."""
    if isinstance(obj, dict):
        # Already a regular dict, just recursively clean values
        return {k: deep_clean(v) for k, v in obj.items()}
    elif hasattr(obj, "items"):
        # Handle dict-like objects (including WandB SummarySubDict)
        # Convert to regular dict first, then recursively clean
        return {k: deep_clean(v) for k, v in dict(obj).items()}
    elif isinstance(obj, (list, tuple)):
        return [deep_clean(v) for v in obj]
    else:
        # For any other type, use clean_numpy_types first
        cleaned = clean_numpy_types(obj)
        # Then verify it's serializable
        json.dumps(cleaned)
        return cleaned
