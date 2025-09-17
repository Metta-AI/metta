"""
W&B utility functions for logging, alerts, and artifact management.
"""

import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import wandb
from wandb import Artifact
from wandb.apis.public.runs import Run
from wandb.errors import CommError

from metta.common.util.constants import METTA_WANDB_ENTITY, METTA_WANDB_PROJECT
from metta.common.util.retry import retry_on_exception
from metta.common.wandb.context import WandbRun
from metta.mettagrid.util.file import WandbURI

logger = logging.getLogger(__name__)

# Global wandb API instance with timeout
wandb_api = wandb.Api(timeout=60)

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


# ============================================================================
# API access functions with retry
# ============================================================================


@wandb_retry
def get_wandb_run(path: str) -> Run:
    """Get wandb run object with retry."""
    return wandb_api.run(path)


@wandb_retry
def get_wandb_artifact(qname: str) -> Artifact:
    """Get wandb artifact with retry."""
    return wandb_api.artifact(qname)


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
# URI utilities
# ============================================================================


def expand_wandb_uri(uri: str, default_project: str = "metta") -> str:
    """Expand short wandb URI formats to full format.

    Handles both short and full wandb URI formats:
    - "wandb://run/my_run_name" ->
      "wandb://ENTITY/metta/model/my_run_name:latest"
      (ENTITY from WANDB_ENTITY or METTA_WANDB_ENTITY)
    - "wandb://run/my_run_name:v5" ->
      "wandb://ENTITY/metta/model/my_run_name:v5"
      (ENTITY from WANDB_ENTITY or METTA_WANDB_ENTITY)
    - "wandb://sweep/sweep_name" ->
      "wandb://ENTITY/metta/sweep_model/sweep_name:latest"
      (ENTITY from WANDB_ENTITY or METTA_WANDB_ENTITY)
    - Full URIs pass through unchanged

    Args:
        uri: Wandb URI to expand
        default_project: Default project name for short URIs

    Returns:
        Expanded wandb URI

    Notes:
        For short URIs (run/..., sweep/...), the entity defaults to
        the current environment `WANDB_ENTITY` or falls back to
        `METTA_WANDB_ENTITY`.
    """
    if not uri.startswith("wandb://"):
        return uri

    path = uri[len("wandb://") :]

    if not path.startswith(("run/", "sweep/")):
        return uri

    # Default entity: respect WANDB_ENTITY if set; otherwise assume METTA_WANDB_ENTITY
    entity = os.getenv("WANDB_ENTITY", METTA_WANDB_ENTITY)

    if path.startswith("run/"):
        run_name = path[4:]
        if ":" in run_name:
            run_name, version = run_name.rsplit(":", 1)
        else:
            version = "latest"
        return f"wandb://{entity}/{default_project}/model/{run_name}:{version}"

    elif path.startswith("sweep/"):
        sweep_name = path[6:]
        if ":" in sweep_name:
            sweep_name, version = sweep_name.rsplit(":", 1)
        else:
            version = "latest"
        return f"wandb://{entity}/{default_project}/sweep_model/{sweep_name}:{version}"

    return uri


# ============================================================================
# Artifact utilities
# ============================================================================


def get_wandb_artifact_metadata(wandb_uri: str) -> Optional[dict]:
    """Extract metadata from a wandb artifact.

    Args:
        wandb_uri: Wandb URI of the artifact

    Returns:
        Artifact metadata dict or None if metadata cannot be extracted
    """
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
    """Upload a file to wandb as an artifact.

    Args:
        file_path: Path to the primary file to upload
        artifact_name: Name for the wandb artifact
        artifact_type: Type of artifact (default: "model")
        metadata: Optional metadata dictionary
        wandb_run: Optional wandb run (uses current run if not provided)
        additional_files: Optional list of additional files to include
        primary_filename: Name for the primary file in the artifact (default: "model.pt")

    Returns:
        Wandb URI of the uploaded artifact or None if upload failed
    """
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


def load_artifact_file(wandb_uri: str, filename: Optional[str] = None, fallback_pattern: str = "*.pt") -> Path:
    """Load a file from wandb artifact.

    Args:
        wandb_uri: Wandb URI (handles both short and full formats)
        filename: Specific file to load from artifact
        fallback_pattern: Pattern to use if filename not found (default: "*.pt")

    Returns:
        Path to the downloaded file

    Raises:
        ValueError: If URI is not a wandb:// URI
        FileNotFoundError: If specified file not found in artifact
    """
    if not wandb_uri.startswith("wandb://"):
        raise ValueError(f"Not a wandb URI: {wandb_uri}")

    expanded_uri = expand_wandb_uri(wandb_uri)
    logger.info(f"Loading artifact from wandb URI: {expanded_uri}")
    uri = WandbURI.parse(expanded_uri)
    qname = uri.qname()

    # Load artifact with retries
    try:
        logger.debug(f"Loading artifact: {qname}")
        artifact = get_wandb_artifact(qname)
    except Exception as e:
        raise ValueError(f"Failed to load artifact: {qname}") from e

    with tempfile.TemporaryDirectory() as temp_dir:
        artifact_dir = Path(temp_dir)

        # Download artifact with retries
        logger.debug(f"Downloading artifact to {artifact_dir}")
        download_artifact(artifact, str(artifact_dir))

        # Find the requested file
        if filename:
            target_file = artifact_dir / filename
            if not target_file.exists():
                raise FileNotFoundError(f"File '{filename}' not found in artifact {wandb_uri}")
        else:
            # Fallback to finding files matching pattern
            matching_files = list(artifact_dir.rglob(fallback_pattern))
            if not matching_files:
                raise FileNotFoundError(f"No files matching '{fallback_pattern}' in artifact {wandb_uri}")
            target_file = matching_files[0]
            if len(matching_files) > 1:
                logger.warning(f"Multiple files found matching '{fallback_pattern}', using: {target_file}")

        # Copy to a persistent location
        import shutil

        persistent_path = Path(tempfile.mktemp(suffix=target_file.suffix))
        shutil.copy2(target_file, persistent_path)
        return persistent_path
