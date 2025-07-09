import logging
import os
import time
from functools import wraps
from typing import Any, Callable, TypeVar

import wandb

logger = logging.getLogger("sweep")

T = TypeVar("T")


def retry_on_exception(
    max_retries: int = 3,
    retry_delay: float = 5.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    logger: logging.Logger | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to retry a function on exception.

    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Delay in seconds between retries
        exceptions: Tuple of exception types to catch and retry on
        logger: Logger instance for logging retry attempts

    Returns:
        Decorated function that implements retry logic
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception | None = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if logger:
                        logger.warning(f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}): {e}")

                    if attempt < max_retries - 1:
                        if logger:
                            logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        if logger:
                            logger.error(f"{func.__name__} failed after {max_retries} attempts")

            # If we get here, all retries failed
            if last_exception is not None:
                raise last_exception
            else:
                # This should never happen, but just in case
                raise RuntimeError(f"{func.__name__} failed without capturing exception")

        return wrapper

    return decorator


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
