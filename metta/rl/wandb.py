"""
RL-specific W&B utilities and metrics setup.
"""

import logging

import torch
import torch.nn as nn

from metta.common.wandb.utils import (
    load_artifact_file,
)
from metta.common.wandb.wandb_context import WandbRun

logger = logging.getLogger(__name__)

# ============================================================================
# RL-specific metric constants
# ============================================================================

POLICY_EVALUATOR_METRIC_PREFIX = "evaluator"
POLICY_EVALUATOR_STEP_METRIC = "metric/evaluator_agent_step"
POLICY_EVALUATOR_EPOCH_METRIC = "metric/evaluator_epoch"


# ============================================================================
# RL-specific metric setup
# ============================================================================


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
    """Set up metrics specific to policy evaluation."""
    # Separate step metric for remote evaluation allows evaluation results to be logged without conflicts
    wandb_run.define_metric(POLICY_EVALUATOR_STEP_METRIC)
    for metric in (f"{POLICY_EVALUATOR_METRIC_PREFIX}/*", f"overview/{POLICY_EVALUATOR_METRIC_PREFIX}/*"):
        wandb_run.define_metric(metric, step_metric=POLICY_EVALUATOR_STEP_METRIC)


# ============================================================================
# Model/policy specific utilities
# ============================================================================


def log_model_parameters(policy: nn.Module, wandb_run: WandbRun) -> None:
    """Log model parameter count to wandb summary."""
    num_params = sum(p.numel() for p in policy.parameters())
    if wandb_run.summary:
        wandb_run.summary["model/total_parameters"] = num_params


def load_policy_from_wandb_uri(wandb_uri: str, device: str | torch.device = "cpu") -> torch.nn.Module:
    """Load policy from wandb URI (handles both short and full formats).

    Accepts:
    - Short format: "wandb://run/my-run" (ENTITY from WANDB_ENTITY or METTA_WANDB_ENTITY)
    - Full format: "wandb://entity/project/artifact:version"

    Args:
        wandb_uri: Wandb URI to load from
        device: Device to load the model to

    Returns:
        Loaded PyTorch model

    Raises:
        ValueError: If URI is not a wandb:// URI
        FileNotFoundError: If no .pt files found in artifact
    """
    if not wandb_uri.startswith("wandb://"):
        raise ValueError(f"Not a wandb URI: {wandb_uri}")

    # Use the common utility to load the artifact file
    logger.info(f"Loading policy from wandb URI: {wandb_uri}")
    model_file = load_artifact_file(wandb_uri, filename="model.pt", fallback_pattern="*.pt")

    try:
        # Load the model
        model = torch.load(model_file, map_location=device, weights_only=False)
        return model
    finally:
        # Clean up the temporary file
        if model_file.exists():
            model_file.unlink()
