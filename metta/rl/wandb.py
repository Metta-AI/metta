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

from metta.common.wandb.wandb_context import WandbRun

logger = logging.getLogger(__name__)

# Use WeakKeyDictionary to associate state with each wandb.Run without mutating the object
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


# Metrics functions moved from metrics.py
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


def log_model_parameters(policy: nn.Module, wandb_run: WandbRun) -> None:
    """Log model parameter count to wandb summary."""
    num_params = sum(p.numel() for p in policy.parameters())
    if wandb_run.summary:
        wandb_run.summary["model/total_parameters"] = num_params


# Policy Loading Functions (moved from wandb_policy_loader.py)


def load_policy_from_wandb_uri(wandb_uri: str, device: str = "cpu") -> Optional[torch.nn.Module]:
    """Load policy from wandb://entity/project/artifact_name:version format.

    Note: This loses filename metadata since wandb artifacts store the file as 'model.pt'.
    The metadata (epoch, agent_step, score, etc.) is stored in the artifact's metadata
    but not in the filename, so we can't use parse_checkpoint_filename() on wandb downloads.
    """
    if not wandb or not wandb_uri.startswith("wandb://"):
        return None

    try:
        artifact_path = wandb_uri[8:]  # Remove "wandb://" prefix
        artifact = wandb.Api().artifact(artifact_path)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = Path(temp_dir)
            artifact.download(root=str(artifact_dir))

            policy_files = list(artifact_dir.rglob("*.pt"))
            if policy_files:
                return torch.load(policy_files[0], map_location=device, weights_only=False)

        return None
    except Exception:
        return None


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
    if not wandb:
        logger.warning("Wandb not available, skipping artifact upload")
        return None

    # Use provided run or get current run
    run = wandb_run or wandb.run
    if run is None:
        logger.warning("No wandb run active, cannot upload artifact")
        return None

    try:
        # Create artifact with metadata
        artifact = wandb.Artifact(name=artifact_name, type=artifact_type, metadata=metadata or {})

        # Add the main checkpoint file - we use a generic name since wandb doesn't preserve paths
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

        qualified_name = artifact.qualified_name
        logger.info(f"Uploaded checkpoint as wandb artifact: {qualified_name}")
        return qualified_name

    except Exception as e:
        logger.error(f"Failed to upload wandb artifact: {e}")
        return None
