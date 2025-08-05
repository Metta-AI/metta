import json
import logging
from typing import Any, List

import wandb
from omegaconf import DictConfig

from metta.common.util.numpy_helpers import clean_numpy_types
from metta.common.wandb.wandb_context import WandbContext

logger = logging.getLogger("sweep")


def create_wandb_run_for_sweep(
    sweep_name: str,
    protein_suggestion: dict[str, Any],
    train_job_cfg: DictConfig,
) -> str | None:
    """
    Create a new wandb run for a sweep using groups instead of W&B sweeps.
    Returns the wandb run ID, or None if WandB initialization failed.
    """

    with WandbContext(train_job_cfg.wandb, train_job_cfg, timeout=120) as wandb_run:
        if wandb_run is None:
            logger.error("Failed to initialize WandB run - WandB may be disabled or connection failed")
            return None

        # Add tags for easy filtering
        if not wandb_run.tags:
            wandb_run.tags = ()
        wandb_run.tags += (f"sweep_name:{sweep_name}", "protein_observation")

        wandb_run.summary.update(
            {
                "protein_suggestion": protein_suggestion,
            }
        )

        # Return the wandb run ID instead of creating dist_cfg.yaml file
        return wandb_run.id


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
    """
    api = wandb.Api()
    wandb_path = f"{wandb_entity}/{wandb_project}"

    # Use the API's native filtering and ordering
    # Order by created_at descending (newest first) and limit results
    runs = api.runs(
        path=wandb_path,
        filters={
            "group": sweep_name,  # Filter by group instead of sweep
            "state": {"$in": ["finished", "failed"]},  # Only get completed runs
            "summary_metrics.protein_observation": {"$exists": True},  # Only runs with observations
        },
        order="-created_at",  # Descending order (newest first)
        per_page=max_observations,  # Limit the number of results
    )

    # Iterate through runs (already filtered and limited)
    return [deep_clean(run.summary.get("protein_observation")) for run in runs]  # type: ignore


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


# 4 - Sweep Analysis Utilities.
# TODO: Move out of here.
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


# TODO: Move out of here.
def sweep_id_from_name(project: str, entity: str, name: str) -> str | None:
    """
    This function is deprecated since we're using groups instead of sweeps.
    Returns the sweep name itself for backward compatibility.

    Args:
        project: WandB project name
        entity: WandB entity name
        name: Sweep name

    Returns:
        The sweep name itself
    """
    logger.warning("sweep_id_from_name is deprecated - using group-based sweeps now")
    return name
