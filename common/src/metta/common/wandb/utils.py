"""
W&B utility functions for logging and alerts.
"""

import logging
import os
from datetime import datetime
from typing import Any

import wandb

from metta.common.util.constants import METTA_WANDB_PROJECT

logger = logging.getLogger(__name__)


def send_wandb_alert(title: str, text: str, run_id: str, project: str, entity: str) -> None:
    """
    Send a W&B alert.

    Args:
        title: Alert title
        text: Alert text/description
        run_id: W&B run ID
        project: W&B project name
        entity: W&B entity/username
    """
    # Validate parameters
    if not all([title, text, run_id, project, entity]):
        raise RuntimeError("All parameters (title, text, run_id, project, entity) are required")

    log_ctx = f"run {entity}/{project}/{run_id}"

    run = wandb.init(
        id=run_id,
        project=project,
        entity=entity,
        resume="must",
        settings=wandb.Settings(init_timeout=15, silent=True, x_disable_stats=True, x_disable_meta=True),
    )
    try:
        run.alert(title=title, text=text)
        logger.info(f"W&B alert '{title}' sent for {log_ctx}")
    finally:
        wandb.finish()


def ensure_wandb_run() -> wandb.Run:
    """
    Ensure a wandb run exists, creating/resuming if needed.
    """
    try:
        import wandb
    except ImportError as e:
        raise RuntimeError("wandb not installed") from e

    # Check if run already exists
    if wandb.run is not None:
        return wandb.run

    # Need to create/resume a run
    run_id = os.environ.get("METTA_RUN_ID")
    if not run_id:
        raise RuntimeError("No active wandb run and METTA_RUN_ID not set")

    # Check if we're in offline mode (no credentials needed)
    wandb_mode = os.environ.get("WANDB_MODE", "").lower()
    if wandb_mode != "offline":
        # Check credentials only if not in offline mode
        api_key = os.environ.get("WANDB_API_KEY")
        has_netrc = os.path.exists(os.path.expanduser("~/.netrc"))

        if not api_key and not has_netrc:
            raise RuntimeError("No wandb credentials (need WANDB_API_KEY or ~/.netrc)")

        # Login if API key provided
        if api_key:
            wandb.login(key=api_key, relogin=True, anonymous="never")

    project = os.environ.get("WANDB_PROJECT", METTA_WANDB_PROJECT)

    # Create/resume run
    run = wandb.init(
        project=project,
        name=run_id,
        id=run_id,
        resume="allow",
        reinit=True,
    )

    # Only print URL if not in offline mode
    if wandb_mode != "offline":
        entity = os.environ.get("WANDB_ENTITY", wandb.api.default_entity)
        logger.info(f"✅ Wandb run: https://wandb.ai/{entity}/{project}/runs/{run_id}")

    return run


def log_to_wandb(metrics: dict[str, Any], step: int = 0, also_summary: bool = True) -> None:
    """
    Log metrics to wandb.

    Args:
        metrics: Dictionary of key-value pairs to log
        step: The step to log at (default 0)
        also_summary: Whether to also add to wandb.summary (default True)
    """
    run = ensure_wandb_run()

    try:
        import wandb

        # Log all metrics
        wandb.log(metrics, step=step)

        # Also add to summary if requested
        if also_summary:
            for key, value in metrics.items():
                run.summary[key] = value

        logger.info(f"✅ Logged {len(metrics)} metrics to wandb")

    except Exception as e:
        raise RuntimeError(f"Failed to log to wandb: {e}") from e


def log_single_value(key: str, value: Any, step: int = 0, also_summary: bool = True) -> None:
    """
    Convenience function to log a single key-value pair.

    Args:
        key: Metric key
        value: Metric value
        step: Step to log at
        also_summary: Whether to add to summary
    """
    log_to_wandb({key: value}, step=step, also_summary=also_summary)


def log_debug_info() -> None:
    """Log various debug information about the environment."""
    debug_metrics = {
        "debug/timestamp": datetime.utcnow().isoformat(),
        "debug/skypilot_task_id": os.environ.get("SKYPILOT_TASK_ID", "not_set"),
        "debug/metta_run_id": os.environ.get("METTA_RUN_ID", "not_set"),
        "debug/wandb_project": os.environ.get("WANDB_PROJECT", "not_set"),
        "debug/hostname": os.environ.get("HOSTNAME", "unknown"),
        "debug/rank": os.environ.get("RANK", "not_set"),
        "debug/local_rank": os.environ.get("LOCAL_RANK", "not_set"),
    }

    logger.info("Debug environment:")
    for k, v in debug_metrics.items():
        logger.info(f"  {k.split('/')[-1]}: {v}")

    log_to_wandb(debug_metrics)
