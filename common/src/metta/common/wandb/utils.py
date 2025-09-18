"""
W&B utility functions for logging, alerts, and artifact management.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import wandb
from wandb import Artifact
from wandb.apis.public.runs import Run
from wandb.errors import CommError

from metta.common.util.retry import retry_on_exception
from metta.common.wandb.context import WandbRun
from metta.mettagrid.util.file import WandbURI

logger = logging.getLogger(__name__)


# Create a custom retry decorator for wandb API calls with sensible defaults
wandb_retry = retry_on_exception(
    max_retries=3,
    initial_delay=2.0,
    max_delay=30.0,
    backoff_factor=2.0,
    exceptions=(CommError, TimeoutError, ConnectionError, OSError),
)


# ============================================================================
# Logging and alerts
# ============================================================================


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


def log_to_wandb(metrics: dict[str, Any], step: int = 0, also_summary: bool = True) -> None:
    """
    Log metrics to wandb.

    Args:
        metrics: Dictionary of key-value pairs to log
        step: The step to log at (default 0)
        also_summary: Whether to also add to wandb.summary (default True)

    """
    if wandb.run is None:
        raise RuntimeError("No active wandb run. Use WandbContext to initialize a run.")

    try:
        # Log all metrics
        wandb.log(metrics, step=step)

        # Also add to summary if requested
        if also_summary:
            for key, value in metrics.items():
                wandb.run.summary[key] = value

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


# ============================================================================
# API access functions with retry
# ============================================================================


@wandb_retry
def get_wandb_run(path: str) -> Run:
    """Get wandb run object with retry."""
    return wandb.Api(timeout=60).run(path)


@wandb_retry
def get_wandb_artifact(qname: str) -> Artifact:
    """Get wandb artifact with retry."""
    return wandb.Api(timeout=60).artifact(qname)


@wandb_retry
def download_artifact(artifact: Artifact, root: str) -> None:
    """Download wandb artifact with retry."""
    artifact.download(root=root)


@wandb_retry
def wait_for_artifact_upload(artifact: Artifact) -> None:
    """Wait for artifact upload to complete with retry."""
    artifact.wait()


# ============================================================================
# Run management utilities
# ============================================================================


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


# ============================================================================
# Artifact utilities
# ============================================================================


def get_wandb_artifact_metadata(wandb_uri: str) -> Optional[dict]:
    """Extract metadata from a wandb artifact."""
    if not wandb_uri.startswith("wandb://"):
        return None

    uri = WandbURI.parse(wandb_uri)

    try:
        artifact = get_wandb_artifact(uri.qname())
        return artifact.metadata
    except Exception as e:
        logger.warning(f"Failed to get artifact metadata for {wandb_uri}: {e}")
        return None


def upload_file_as_artifact(
    file_path: str,
    artifact_name: str,
    artifact_type: str = "model",
    metadata: Optional[dict] = None,
    wandb_run: Optional[WandbRun] = None,
    additional_files: Optional[list[str]] = None,
    primary_filename: str = "model.pt",
) -> Optional[str]:
    """Upload a file to wandb as an artifact."""
    # Use provided run or get current run
    run = wandb_run or wandb.run
    if run is None:
        logger.warning("No wandb run active, cannot upload artifact")
        return None

    # Prepare metadata
    artifact_metadata = metadata.copy() if metadata else {}

    # Create artifact (wandb supports dots in names)
    artifact = wandb.Artifact(name=artifact_name, type=artifact_type, metadata=artifact_metadata)

    # Add primary file
    artifact.add_file(file_path, name=primary_filename)

    # Add any additional files
    if additional_files:
        for file_path in additional_files:
            if Path(file_path).exists():
                artifact.add_file(file_path)
            else:
                logger.warning(f"Additional file not found: {file_path}")

    # Log artifact to run
    run.log_artifact(artifact)

    # Wait for upload to complete with retries
    try:
        wait_for_artifact_upload(artifact)
    except Exception as e:
        logger.error(f"Failed to wait for artifact upload: {e}")
        # Even if wait fails, the artifact might still upload in the background

    wandb_uri = f"wandb://{run.project}/{artifact_name}:{artifact.version}"
    logger.info(f"Uploaded file as wandb artifact: {artifact.qualified_name}")

    return wandb_uri
