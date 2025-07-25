import json
import logging
import time
from typing import Any, List

import wandb
from omegaconf import DictConfig, OmegaConf

from metta.common.util.numpy_helpers import clean_numpy_types
from metta.common.util.retry import retry_on_exception
from metta.common.wandb.wandb_context import WandbContext

logger = logging.getLogger("sweep")


@retry_on_exception(max_retries=3, retry_delay=5.0, logger=logger)
def _fetch_sweep_from_api(project: str, entity: str, name: str) -> str | None:
    """
    Fetch sweep ID from WandB API by name.

    This is the core business logic separated from retry logic.

    Args:
        project: WandB project name
        entity: WandB entity name
        name: Sweep name to search for

    Returns:
        Sweep ID if found, None otherwise
    """
    logger.info(f"Attempting to fetch sweep '{name}'")

    api = wandb.Api()
    start_time = time.time()

    # Get project first to check if it exists
    try:
        proj = api.project(project, entity)
    except Exception as e:
        logger.error(f"Failed to access project {entity}/{project}: {e}")
        return None

    # Fetch sweeps with a generator to avoid loading all at once
    sweeps = proj.sweeps()

    found_sweep_id = None
    sweep_count = 0

    # Process sweeps one by one to find the matching name
    for sweep in sweeps:
        sweep_count += 1
        if sweep.name == name:
            found_sweep_id = sweep.id
            logger.info(
                f"Found existing sweep: {name} with ID: {found_sweep_id} "
                f"(checked {sweep_count} sweeps in {time.time() - start_time:.2f}s)"
            )
            return found_sweep_id

        # Log progress periodically
        if sweep_count % 10 == 0:
            logger.debug(f"Checked {sweep_count} sweeps so far...")

    elapsed = time.time() - start_time
    logger.info(f"No existing sweep found with name: {name} (checked {sweep_count} sweeps in {elapsed:.2f}s)")
    return None


def create_wandb_sweep(
    wandb_entity: str,
    wandb_project: str,
    wandb_sweep_name: str,
) -> str:
    """
    Create a new wandb sweep with parameters from Protein sweep configuration.

    Args:
        wandb_entity (str): The wandb entity (username or team name).
        wandb_project (str): The wandb project name.
        wandb_sweep_name (str): The name of the sweep.

    Returns:
        str: The ID of the created sweep.
    """
    wandb_parameters = {"dummy_param": {"values": [1]}}

    logger.info(f"Creating WandB sweep '{wandb_sweep_name}'")

    sweep_id = wandb.sweep(
        sweep={
            "name": wandb_sweep_name,
            "method": "random",  # Dummy method
            "parameters": wandb_parameters,  # Dummy parameters
        },
        project=wandb_project,
        entity=wandb_entity,
    )
    logger.info(f"Created WandB sweep '{wandb_sweep_name}' with ID: {sweep_id}")
    return sweep_id


def create_wandb_run_for_sweep(
    wandb_sweep_id: str,
    wandb_entity: str,
    wandb_project: str,
    sweep_name: str,
    run_name: str,
    protein_suggestion: dict[str, Any],
    cfg: DictConfig,
) -> None:
    """
    Create a new wandb run for a sweep.
    """

    # This function is passed to agent so we can immediately
    # modify the run's name, tags, and save the protein suggestion.
    def init_run():
        with WandbContext(cfg.wandb, cfg) as wandb_run:
            if wandb_run is None:
                logger.error("Failed to initialize WandB run - WandB may be disabled or connection failed")
                return

            wandb_run.name = run_name

            if not wandb_run.tags:
                wandb_run.tags = ()
            wandb_run.tags += (f"sweep_id:{wandb_sweep_id}", f"sweep_name:{sweep_name}")

            wandb_run.summary.update(
                {
                    "protein_suggestion": protein_suggestion,
                }
            )

            OmegaConf.save(
                {
                    "run": run_name,  # Your custom name
                    "wandb_run_id": wandb_run.id,  # The agent-generated ID
                },
                cfg.dist_cfg_path,
            )

    # WandB agent MUST be used to associate the new run with the sweep.
    wandb.agent(
        wandb_sweep_id,
        entity=wandb_entity,
        project=wandb_project,
        function=init_run,
        count=1,
    )


# 2 - Protein Integration Utilities.
def fetch_protein_observations_from_wandb(
    wandb_entity: str,
    wandb_project: str,
    wandb_sweep_id: str,
    max_observations: int = 100,
) -> list[dict[str, Any]]:
    """
    Fetch latest protein observations from WandB sweep runs.

    Args:
        entity: The WandB entity name.
        project: The WandB project name.
        wandb_sweep_id: The WandB sweep ID.
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
            "sweep": wandb_sweep_id,
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
def get_sweep_runs(sweep_id: str, entity: str, project: str) -> List[Any]:
    """Get all runs from a sweep sorted by score."""
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")

    # Get all runs and filter for successful ones
    runs = []
    for run in sweep.runs:
        if run.summary.get("protein.state") == "success":
            score = run.summary.get("score", run.summary.get("protein.objective", 0))
            if score is not None and score > 0:  # Filter out failed runs
                runs.append(run)

    # Sort by score (descending for reward metric)
    runs.sort(key=lambda r: r.summary.get("score", r.summary.get("protein.objective", 0)), reverse=True)
    return runs


# TODO: Move out of here.
def sweep_id_from_name(project: str, entity: str, name: str) -> str | None:
    """
    Get sweep ID from name with retry logic for network issues.

    Args:
        project: WandB project name
        entity: WandB entity name
        name: Sweep name to search for

    Returns:
        Sweep ID if found, None otherwise
    """
    try:
        return _fetch_sweep_from_api(project, entity, name)
    except Exception as e:
        logger.error(f"Failed to fetch sweeps after all retry attempts. Assuming no existing sweep. Error: {e}")
        # Return None to allow sweep creation to proceed
        return None
