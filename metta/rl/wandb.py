"""RL-specific W&B utilities and metrics setup."""

import logging

import torch.nn as nn
from torch.nn.parameter import UninitializedParameter

from metta.common.wandb.context import WandbRun

logger = logging.getLogger(__name__)


POLICY_EVALUATOR_METRIC_PREFIX = "evaluator"
POLICY_EVALUATOR_STEP_METRIC = "metric/evaluator_agent_step"
POLICY_EVALUATOR_EPOCH_METRIC = "metric/evaluator_epoch"


def setup_wandb_metrics(wandb_run: WandbRun) -> None:
    """Set up wandb metric definitions for consistent tracking across runs."""
    metrics = ["agent_step", "epoch", "total_time", "train_time"]
    for metric in metrics:
        wandb_run.define_metric(f"metric/{metric}")

    wandb_run.define_metric("*", step_metric="metric/agent_step")
    wandb_run.define_metric("overview/reward_vs_total_time", step_metric="metric/total_time")
    setup_policy_evaluator_metrics(wandb_run)


def setup_policy_evaluator_metrics(wandb_run: WandbRun) -> None:
    # Separate step metric for remote evaluation allows evaluation results to be logged without conflicts
    """Set up metrics specific to policy evaluation."""
    wandb_run.define_metric(POLICY_EVALUATOR_STEP_METRIC)
    for metric in (f"{POLICY_EVALUATOR_METRIC_PREFIX}/*", f"overview/{POLICY_EVALUATOR_METRIC_PREFIX}/*"):
        wandb_run.define_metric(metric, step_metric=POLICY_EVALUATOR_STEP_METRIC)


def log_model_parameters(policy: nn.Module, wandb_run: WandbRun) -> None:
    """Log model parameter count to wandb summary."""
    params = list(policy.parameters())
    skipped_lazy_params = sum(isinstance(param, UninitializedParameter) for param in params)
    num_params = sum(param.numel() for param in params if not isinstance(param, UninitializedParameter))

    if skipped_lazy_params:
        logger.debug(
            "Skipped %d uninitialized parameters when logging model size.",
            skipped_lazy_params,
        )

    if wandb_run.summary:
        wandb_run.summary["model/total_parameters"] = num_params
