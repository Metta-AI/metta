"""RL-specific W&B utilities and metrics setup."""

import logging

import torch.nn as nn
from torch.nn.parameter import UninitializedParameter

from metta.common.wandb.context import WandbRun

logger = logging.getLogger(__name__)


TRAIN_AGENT_STEP_METRIC = "metric/agent_step"
TRAIN_EPOCH_METRIC = "metric/epoch"
TRAIN_TOTAL_TIME_METRIC = "metric/total_time"
TRAIN_TRAIN_TIME_METRIC = "metric/train_time"

POLICY_EVALUATOR_METRIC_PREFIX = "evaluator"
POLICY_EVALUATOR_STEP_METRIC = "metric/evaluator_agent_step"
POLICY_EVALUATOR_EPOCH_METRIC = "metric/evaluator_epoch"


def build_step_metrics(
    agent_step: int,
    epoch: int,
    *,
    step_metric: str,
    epoch_metric: str,
) -> dict[str, float]:
    return {step_metric: float(agent_step), epoch_metric: float(epoch)}


def build_training_step_metrics(agent_step: int, epoch: int) -> dict[str, float]:
    return build_step_metrics(
        agent_step=agent_step,
        epoch=epoch,
        step_metric=TRAIN_AGENT_STEP_METRIC,
        epoch_metric=TRAIN_EPOCH_METRIC,
    )


def build_evaluator_step_metrics(agent_step: int, epoch: int) -> dict[str, float]:
    return build_step_metrics(
        agent_step=agent_step,
        epoch=epoch,
        step_metric=POLICY_EVALUATOR_STEP_METRIC,
        epoch_metric=POLICY_EVALUATOR_EPOCH_METRIC,
    )


def setup_wandb_metrics(wandb_run: WandbRun) -> None:
    """Set up wandb metric definitions for consistent tracking across runs."""
    for metric in (
        TRAIN_AGENT_STEP_METRIC,
        TRAIN_EPOCH_METRIC,
        TRAIN_TOTAL_TIME_METRIC,
        TRAIN_TRAIN_TIME_METRIC,
    ):
        wandb_run.define_metric(metric)

    wandb_run.define_metric("*", step_metric=TRAIN_AGENT_STEP_METRIC)
    wandb_run.define_metric("overview/reward_vs_total_time", step_metric=TRAIN_TOTAL_TIME_METRIC)
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
