import logging
import os
import time

import wandb

from metta.common.util.retry import retry_on_exception

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


def create_wandb_sweep(sweep_name: str, wandb_entity: str, wandb_project: str) -> str:
    """
    Create a new wandb sweep with a dummy parameter (Protein will control all suggestions).

    Args:
        sweep_name (str): The name of the sweep.
        wandb_entity (str): The wandb entity (username or team name).
        wandb_project (str): The wandb project name.

    Returns:
        str: The ID of the created sweep.
    """
    sweep_id = wandb.sweep(
        sweep={
            "name": sweep_name,
            "method": "bayes",  # This won't actually be used since we override suggestions
            "metric": {"name": "protein.objective", "goal": "maximize"},
            "parameters": {
                # WandB requires at least one parameter, but Protein will override all suggestions
                "dummy_param": {"values": [1]}
            },
        },
        project=wandb_project,
        entity=wandb_entity,
    )
    return sweep_id


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


def generate_run_id_for_sweep(sweep_id: str, sweep_runs_dir: str) -> str:
    api = wandb.Api()
    sweep = api.sweep(sweep_id)

    used_ids = set()
    used_names = set(run.name for run in sweep.runs).union(set(os.listdir(sweep_runs_dir)))
    for name in used_names:
        try:
            id = int(name.split(".")[-1])
            used_ids.add(id)
        except ValueError:
            logger.warning(f"Invalid run name: {name}, not ending with an integer")

    id = 0
    if len(used_ids) > 0:
        id = max(used_ids) + 1

    return f"{sweep.name}.r.{id}"
