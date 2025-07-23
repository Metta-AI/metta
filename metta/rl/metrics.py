"""Metrics setup and logging utilities for wandb."""

from typing import Any, Optional

import torch.nn as nn


def setup_wandb_metrics(wandb_run: Any) -> None:
    """Set up wandb metric definitions for consistent tracking across runs.

    Args:
        wandb_run: The wandb run object
    """
    # Define base metrics
    metrics = ["agent_step", "epoch", "total_time", "train_time"]
    for metric in metrics:
        wandb_run.define_metric(f"metric/{metric}")

    # Set agent_step as the default x-axis for all metrics
    wandb_run.define_metric("*", step_metric="metric/agent_step")

    # Define special metric for reward vs total time
    wandb_run.define_metric("overview/reward_vs_total_time", step_metric="metric/total_time")


def log_model_parameters(policy: nn.Module, wandb_run: Any) -> None:
    """Log model parameter count to wandb summary.

    Args:
        policy: The policy model
        wandb_run: The wandb run object
    """
    num_params = sum(p.numel() for p in policy.parameters())
    if wandb_run.summary:
        wandb_run.summary["model/total_parameters"] = num_params


def setup_wandb_metrics_and_log_model(
    policy: nn.Module,
    wandb_run: Optional[Any],
    is_master: bool = True,
) -> None:
    """Convenience function to set up metrics and log model parameters.

    Args:
        policy: The policy model
        wandb_run: The wandb run object (optional)
        is_master: Whether this is the master process in distributed training
    """
    if wandb_run and is_master:
        setup_wandb_metrics(wandb_run)
        log_model_parameters(policy, wandb_run)


def log_training_metrics(
    wandb_run: Any,
    metrics: dict[str, Any],
    step: int,
) -> None:
    """Log training metrics to wandb.

    Args:
        wandb_run: The wandb run object
        metrics: Dictionary of metrics to log
        step: The current training step
    """
    wandb_run.log(metrics, step=step)


def define_custom_metric(
    wandb_run: Any,
    metric_name: str,
    step_metric: Optional[str] = None,
) -> None:
    """Define a custom metric with optional step metric.

    Args:
        wandb_run: The wandb run object
        metric_name: Name of the metric to define
        step_metric: Optional step metric to use as x-axis
    """
    if step_metric:
        wandb_run.define_metric(metric_name, step_metric=step_metric)
    else:
        wandb_run.define_metric(metric_name)
