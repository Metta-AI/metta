"""
W&B utility functions for logging and alerts.
"""

import logging

import wandb
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter
from wandb.apis.public.runs import Run
from wandb.errors import CommError

from metta.common.wandb.context import WandbRun

logger = logging.getLogger(__name__)


# Create a custom retry decorator for wandb API calls with sensible defaults
wandb_retry = retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential_jitter(initial=2.0, max=30.0, exp_base=2.0),
    retry=retry_if_exception_type((CommError, TimeoutError, ConnectionError, OSError)),
    reraise=True,
)


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
