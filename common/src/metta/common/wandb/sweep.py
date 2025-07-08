import logging
import os
import time

import wandb

logger = logging.getLogger("sweep")


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
    api = wandb.Api()

    # Retry logic for network issues
    max_retries = 3
    retry_delay = 5  # seconds

    # Robust WandB API call with retry logic for stability
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting to fetch sweep '{name}' (attempt {attempt + 1}/{max_retries})")

            # OPTIMIZATION: Try to fetch sweeps more efficiently
            # Unfortunately, WandB API doesn't support filtering sweeps by name directly
            # But we can try to limit the number of sweeps fetched
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

        except Exception as e:
            logger.warning(f"Network error fetching sweeps (attempt {attempt + 1}/{max_retries}): {e}")

            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            else:
                logger.error(f"Failed to fetch sweeps after {max_retries} attempts. Assuming no existing sweep.")
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
