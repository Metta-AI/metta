"""Optimization utilities for Metta training."""

import logging
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)


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
