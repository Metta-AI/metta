"""Optimization utilities for Metta training."""

import logging
from typing import Any, Dict

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def calculate_explained_variance(values: Tensor, advantages: Tensor) -> float:
    """Calculate explained variance for value function evaluation."""
    y_pred = values.flatten()
    y_true = advantages.flatten() + values.flatten()
    var_y = y_true.var()
    explained_var = torch.nan if var_y == 0 else 1 - (y_true - y_pred).var() / var_y
    return explained_var.item() if torch.is_tensor(explained_var) else float("nan")


def compute_gradient_stats(policy: torch.nn.Module) -> Dict[str, float]:
    """Compute gradient statistics for the policy.

    Returns:
        Dictionary with 'grad/mean', 'grad/variance', and 'grad/norm' keys
    """
    all_gradients = []
    for param in policy.parameters():
        if param.grad is not None:
            all_gradients.append(param.grad.view(-1))

    if not all_gradients:
        return {}

    all_gradients_tensor = torch.cat(all_gradients).to(torch.float32)

    grad_mean = all_gradients_tensor.mean()
    grad_variance = all_gradients_tensor.var()
    grad_norm = all_gradients_tensor.norm(2)

    grad_stats = {
        "grad/mean": grad_mean.item(),
        "grad/variance": grad_variance.item(),
        "grad/norm": grad_norm.item(),
    }

    return grad_stats


def maybe_update_l2_weights(
    agent: Any,
    epoch: int,
    interval: int,
    is_master: bool = True,
    force: bool = False,
) -> None:
    """Update L2 weights if on interval.

    Args:
        agent: Policy/agent with update_l2_init_weight_copy method
        epoch: Current epoch
        interval: Update interval (0 to disable)
        is_master: Whether this is the master process
        force: Force update regardless of interval
    """
    if not is_master or not interval:
        return

    if force or epoch % interval == 0:
        if hasattr(agent, "update_l2_init_weight_copy"):
            agent.update_l2_init_weight_copy()
            logger.info(f"Updated L2 init weight copy at epoch {epoch}")
