"""
W&B utility functions for logging and alerts.
"""

import logging
from typing import Any

import wandb
from wandb.apis.public.runs import Run
from wandb.errors import CommError

from metta.common.util.retry import retry_on_exception
from metta.common.wandb.context import WandbRun

logger = logging.getLogger(__name__)


# Create a custom retry decorator for wandb API calls with sensible defaults
wandb_retry = retry_on_exception(
    max_retries=3,
    initial_delay=2.0,
    max_delay=30.0,
    backoff_factor=2.0,
    exceptions=(CommError, TimeoutError, ConnectionError, OSError),
)


def send_wandb_alert(title: str, text: str, run_id: str, project: str, entity: str) -> None:
    if not all([title, text, run_id, project, entity]):
        raise RuntimeError("All parameters (title, text, run_id, project, entity) are required")

    log_ctx = f"run {entity}/{project}/{run_id}"

    run = wandb.init(
        id=run_id,
        project=project,
        entity=entity,
        resume="allow",
        settings=wandb.Settings(init_timeout=15, silent=True, x_disable_stats=True, x_disable_meta=True),
    )
    try:
        run.alert(title=title, text=text)
        logger.info(f"W&B alert '{title}' sent for {log_ctx}")
    finally:
        wandb.finish()


def log_to_wandb_summary(data: dict[str, Any]) -> None:
    """Log key-value pairs to wandb summary for cross-run comparison."""
    if wandb.run is None:
        raise RuntimeError("No active wandb run. Use WandbContext to initialize a run.")

    try:
        # Simply update the summary with all key-value pairs
        for key, value in data.items():
            wandb.run.summary[key] = value

        logger.info(f"✅ Added {len(data)} items to wandb summary")

    except Exception as e:
        raise RuntimeError(f"Failed to log to wandb summary: {e}") from e


@wandb_retry
def get_wandb_run(path: str) -> Run:
    """Get wandb run object with retry."""
    return wandb.Api(timeout=60).run(path)


def abort_requested(wandb_run: WandbRun | None) -> bool:
    """Check if wandb run has an 'abort' tag."""
    if wandb_run is None:
        return False

    try:
        run_obj = get_wandb_run(wandb_run.path)
        has_abort = "abort" in run_obj.tags
        if has_abort:
            logger.info(f"Abort tag found on run {wandb_run.path}")
        return has_abort
    except Exception as e:
        logger.debug(f"Abort tag check failed: {e}")
        # Don't abort on API errors - let training continue
        return False
