"""
Optimized AGaLiTe layers with vectorized operations for better performance.
This replaces the loop-based discounted sum with efficient parallel operations.
"""

import torch

from metta.ops import agalite_kernels


@torch.jit.script
def _jit_discounted_sum(start_state: torch.Tensor, x: torch.Tensor, discounts: torch.Tensor) -> torch.Tensor:
    """JIT-compiled discounted sum that avoids inplace operations.

    This version builds the output sequentially without modifying existing tensors,
    which is essential for autograd compatibility.
    """
    T = x.shape[0]
    if T == 0:
        return x

    # Build outputs list - this avoids inplace operations
    outputs = []

    # Initialize with first step
    current = discounts[0] * start_state + x[0]
    outputs.append(current)

    # Process remaining timesteps
    for t in range(1, T):
        # Create new tensor for each timestep (no inplace modification)
        current = discounts[t] * current + x[t]
        outputs.append(current)

    # Stack all outputs into final tensor
    return torch.stack(outputs, dim=0)


def _python_discounted_sum(start_state: torch.Tensor, x: torch.Tensor, discounts: torch.Tensor) -> torch.Tensor:
    """Optimized discounted sum with minimal overhead."""
    # Ensure start_state has same shape as x[0] - do this efficiently
    if start_state.dim() < x.dim() - 1:
        shape_diff = x.dim() - 1 - start_state.dim()
        for _ in range(shape_diff):
            start_state = start_state.unsqueeze(-1)

    return _jit_discounted_sum(start_state, x, discounts)


def discounted_sum(start_state: torch.Tensor, x: torch.Tensor, discounts: torch.Tensor) -> torch.Tensor:
    """Compute discounted sums using fused kernels when available."""

    try:
        return agalite_kernels.fused_discounted_sum(start_state, x, discounts)
    except RuntimeError:
        return _python_discounted_sum(start_state, x, discounts)


# Alias for batched version (it's the same operation)
batched_discounted_sum = discounted_sum


__all__ = ["discounted_sum", "batched_discounted_sum", "_python_discounted_sum"]
