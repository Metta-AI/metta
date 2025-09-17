from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import wandb
from wandb import Artifact
from wandb.apis.public.runs import Run
from wandb.errors import CommError

from metta.common.util.constants import METTA_WANDB_ENTITY
from metta.common.util.retry import retry_on_exception
from metta.common.wandb.wandb_context import WandbRun
from metta.mettagrid.util.file import WandbURI

logger = logging.getLogger(__name__)
wandb_api = wandb.Api(timeout=60)

# Create a custom retry decorator for wandb API calls with sensible defaults
wandb_retry = retry_on_exception(
    max_retries=3,
    initial_delay=2.0,
    max_delay=30.0,
    backoff_factor=2.0,
    exceptions=(CommError, TimeoutError, ConnectionError, OSError),
)


@wandb_retry
def _get_wandb_run(path: str) -> Run:
    """Get wandb run object with retry."""
    return wandb_api.run(path)


@wandb_retry
def _get_wandb_artifact(qname: str) -> Artifact:
    """Get wandb artifact with retry."""
    return wandb_api.artifact(qname)


@wandb_retry
def _download_artifact(artifact: Artifact, root: str) -> None:
    """Download wandb artifact with retry."""
    artifact.download(root=root)


@wandb_retry
def _wait_for_artifact_upload(artifact: Artifact) -> None:
    """Wait for artifact upload to complete with retry."""
    artifact.wait()


def abort_requested(wandb_run: WandbRun | None) -> bool:
    """Check if wandb run has an 'abort' tag.

    Used for graceful early stopping of training runs. When an 'abort' tag is added
    to a wandb run, the training loop will complete its current epoch and then stop,
    updating the total_timesteps to reflect the actual steps completed.

    Args:
        wandb_run: The wandb run to check
        min_interval_sec: Kept for backward compatibility (no longer used)

    Returns:
        True if the run has an 'abort' tag, False otherwise
    """
    if wandb_run is None:
        return False

    try:
        run_obj = _get_wandb_run(wandb_run.path)
        has_abort = "abort" in run_obj.tags
        if has_abort:
            logger.info(f"Abort tag found on run {wandb_run.path}")
        return has_abort
    except Exception as e:
        logger.debug(f"Abort tag check failed: {e}")
        # Don't abort on API errors - let training continue
        return False


POLICY_EVALUATOR_METRIC_PREFIX = "evaluator"
POLICY_EVALUATOR_STEP_METRIC = "metric/evaluator_agent_step"
POLICY_EVALUATOR_EPOCH_METRIC = "metric/evaluator_epoch"


def setup_wandb_metrics(wandb_run: WandbRun) -> None:
    """Set up wandb metric definitions for consistent tracking across runs."""
    # Define base metrics
    metrics = ["agent_step", "epoch", "total_time", "train_time"]
    for metric in metrics:
        wandb_run.define_metric(f"metric/{metric}")

    # Set agent_step as the default x-axis for all metrics
    wandb_run.define_metric("*", step_metric="metric/agent_step")

    # Define special metric for reward vs total time
    wandb_run.define_metric("overview/reward_vs_total_time", step_metric="metric/total_time")
    setup_policy_evaluator_metrics(wandb_run)


def setup_policy_evaluator_metrics(wandb_run: WandbRun) -> None:
    # Separate step metric for remote evaluation allows evaluation results to be logged without conflicts
    wandb_run.define_metric(POLICY_EVALUATOR_STEP_METRIC)
    for metric in (f"{POLICY_EVALUATOR_METRIC_PREFIX}/*", f"overview/{POLICY_EVALUATOR_METRIC_PREFIX}/*"):
        wandb_run.define_metric(metric, step_metric=POLICY_EVALUATOR_STEP_METRIC)


def log_model_parameters(policy: nn.Module, wandb_run: WandbRun) -> None:
    """Log model parameter count to wandb summary."""
    num_params = sum(p.numel() for p in policy.parameters())
    if wandb_run.summary:
        wandb_run.summary["model/total_parameters"] = num_params


def get_wandb_checkpoint_metadata(wandb_uri: str) -> Optional[dict]:
    """Extract checkpoint metadata from a wandb artifact.

    Returns a dict with keys: run_name, epoch, agent_step, total_time, score
    or None if metadata cannot be extracted.
    """
    if not wandb_uri.startswith("wandb://"):
        return None

    uri = WandbURI.parse(wandb_uri)

    try:
        artifact = _get_wandb_artifact(uri.qname())
        metadata = artifact.metadata
    except Exception as e:
        logger.warning(f"Failed to get artifact metadata for {wandb_uri}: {e}")
        return None

    if metadata is None:
        return None

    # Check if we have all required fields
    required_fields = ["run_name", "epoch", "agent_step", "total_time", "score"]
    if all(field in metadata for field in required_fields):
        return {
            "run_name": f"{metadata['run_name']}:{artifact.version}",
            "epoch": metadata["epoch"],
            "agent_step": metadata["agent_step"],
            "total_time": metadata["total_time"],
            "score": metadata["score"],
        }
    return None


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


def load_policy_from_wandb_uri(wandb_uri: str, device: str | torch.device = "cpu") -> torch.nn.Module:
    """Load policy from wandb URI (handles both short and full formats).

    Accepts:
    - Short format: "wandb://run/my-run" (ENTITY from WANDB_ENTITY or METTA_WANDB_ENTITY)
    - Full format: "wandb://entity/project/artifact:version"

    Raises:
        ValueError: If URI is not a wandb:// URI
        FileNotFoundError: If no .pt files found in artifact
    """
    if not wandb_uri.startswith("wandb://"):
        raise ValueError(f"Not a wandb URI: {wandb_uri}")

    expanded_uri = expand_wandb_uri(wandb_uri)
    logger.info(f"Loading policy from wandb URI: {expanded_uri}")
    uri = WandbURI.parse(expanded_uri)
    qname = uri.qname()

    # Load artifact with retries
    try:
        logger.debug(f"Loading artifact: {qname}")
        artifact = _get_wandb_artifact(qname)
    except Exception as e:
        raise ValueError(f"Failed to load artifact: {qname}") from e

    with tempfile.TemporaryDirectory() as temp_dir:
        artifact_dir = Path(temp_dir)

        # Download artifact with retries
        logger.debug(f"Downloading artifact to {artifact_dir}")
        _download_artifact(artifact, str(artifact_dir))

        # Load model.pt
        model_file = artifact_dir / "model.pt"
        if not model_file.exists():
            # Fallback to any .pt file
            policy_files = list(artifact_dir.rglob("*.pt"))
            if not policy_files:
                raise FileNotFoundError(f"No .pt files in artifact {wandb_uri}")
            model_file = policy_files[0]

        return torch.load(model_file, map_location=device, weights_only=False)


def upload_checkpoint_as_artifact(
    checkpoint_path: str,
    artifact_name: str,
    artifact_type: str = "model",
    metadata: Optional[dict] = None,
    wandb_run: Optional[WandbRun] = None,
    additional_files: Optional[list[str]] = None,
) -> Optional[str]:
    """Upload a checkpoint file to wandb as an artifact."""
    # Use provided run or get current run
    run = wandb_run or wandb.run
    if run is None:
        logger.warning("No wandb run active, cannot upload artifact")
        return None

    # Prepare metadata
    artifact_metadata = metadata.copy() if metadata else {}

    # Create artifact (wandb supports dots in names)
    artifact = wandb.Artifact(name=artifact_name, type=artifact_type, metadata=artifact_metadata)

    # Add checkpoint file as model.pt
    artifact.add_file(checkpoint_path, name="model.pt")

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
        _wait_for_artifact_upload(artifact)
    except Exception as e:
        logger.error(f"Failed to wait for artifact upload: {e}")
        # Even if wait fails, the artifact might still upload in the background

    wandb_uri = f"wandb://{run.project}/{artifact_name}:{artifact.version}"
    logger.info(f"Uploaded checkpoint as wandb artifact: {artifact.qualified_name}")

    return wandb_uri
