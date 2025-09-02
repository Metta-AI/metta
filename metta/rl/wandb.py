from __future__ import annotations

import logging
import tempfile
import time
import weakref
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import wandb
from wandb import Artifact

from metta.common.wandb.wandb_context import WandbRun
from metta.mettagrid.util.file import WandbURI

logger = logging.getLogger(__name__)

_ABORT_STATE: weakref.WeakKeyDictionary[WandbRun, Dict[str, float | bool]] = weakref.WeakKeyDictionary()


def abort_requested(wandb_run: WandbRun | None, min_interval_sec: int = 60) -> bool:
    """Check if wandb run has an 'abort' tag, throttling API calls to min_interval_sec."""
    if wandb_run is None:
        return False

    state = _ABORT_STATE.setdefault(wandb_run, {"last_check": 0.0, "cached_result": False})
    now = time.time()

    # Return cached result if within throttle interval
    if now - state["last_check"] < min_interval_sec:
        return bool(state["cached_result"])

    # Time to check again
    state["last_check"] = now
    try:
        run_obj = wandb.Api().run(wandb_run.path)
        state["cached_result"] = "abort" in run_obj.tags
    except Exception as e:
        logger.debug(f"Abort tag check failed: {e}")
        state["cached_result"] = False

    return bool(state["cached_result"])


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
    artifact: Artifact = wandb.Api().artifact(uri.qname())
    metadata = artifact.metadata

    if metadata is None:
        return None

    # Check if we have all required fields
    required_fields = ["run_name", "epoch", "agent_step", "total_time", "score"]
    if all(field in metadata for field in required_fields):
        return {
            "run_name": metadata["run_name"],
            "epoch": metadata["epoch"],
            "agent_step": metadata["agent_step"],
            "total_time": metadata["total_time"],
            "score": metadata["score"],
        }
    return None


def load_policy_from_wandb_uri(wandb_uri: str, device: str | torch.device = "cpu") -> torch.nn.Module:
    """Load policy from wandb://entity/project/artifact_name:version format.
    Raises:
        ValueError: If URI is not a wandb:// URI
        FileNotFoundError: If no .pt files found in artifact
    """
    if not wandb_uri.startswith("wandb://"):
        raise ValueError(f"Not a wandb URI: {wandb_uri}")

    logger.info(f"Loading policy from wandb URI: {wandb_uri}")
    uri = WandbURI.parse(wandb_uri)
    artifact = wandb.Api().artifact(uri.qname())

    with tempfile.TemporaryDirectory() as temp_dir:
        artifact_dir = Path(temp_dir)
        artifact.download(root=str(artifact_dir))

        # Look for model.pt file (our standard name)
        model_file = artifact_dir / "model.pt"
        if not model_file.exists():
            # Fallback to any .pt file
            policy_files = list(artifact_dir.rglob("*.pt"))
            if not policy_files:
                raise FileNotFoundError(f"No .pt files found in wandb artifact {wandb_uri}")
            model_file = policy_files[0]
            logger.warning(f"model.pt not found, using {model_file.name}")

        # Load the policy
        return torch.load(model_file, map_location=device, weights_only=False)


# Minimal Wandb Artifact Upload Functions


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

    # Prepare metadata with original filename
    artifact_metadata = metadata.copy() if metadata else {}
    artifact_metadata["original_filename"] = Path(checkpoint_path).name

    # Create artifact with complete metadata
    artifact = wandb.Artifact(name=artifact_name, type=artifact_type, metadata=artifact_metadata)

    # Add the main checkpoint file - we use a generic name for consistency
    # The actual metadata is stored in the artifact's metadata field
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

    # Wait for upload to complete
    artifact.wait()

    # Always use "latest" for simplicity and reliability
    # This avoids all timing issues with WandB version assignment
    wandb_uri = f"wandb://{run.project}/{artifact_name}:latest"

    # Log the actual qualified_name for debugging
    logger.info(f"Uploaded checkpoint as wandb artifact: {artifact.qualified_name}")
    logger.debug(f"Returning simplified WandB URI: {wandb_uri}")

    return wandb_uri
