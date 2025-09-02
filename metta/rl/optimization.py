"""Optimization utilities for Metta training."""

import logging

import torch

from metta.agent.metta_agent import PolicyAgent

logger = logging.getLogger(__name__)


def compute_gradient_stats(policy: PolicyAgent) -> dict[str, float]:
    """Compute gradient mean, variance, and norm statistics for the policy."""
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
