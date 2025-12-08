"""Utilities for handling unused parameters in DDP training."""

from typing import Sequence

from tensordict import TensorDict
from torch import Tensor


def add_dummy_loss_for_unused_params(
    loss: Tensor,
    *,
    td: TensorDict,
    used_keys: Sequence[str],
) -> Tensor:
    """Add zero-weighted terms to loss for unused TensorDict outputs to satisfy DDP.

    PyTorch's DistributedDataParallel (DDP) requires all model parameters to participate
    in every backward pass. When a loss function only uses some policy outputs (e.g.,
    `act_log_prob` but not `values`), the parameters that produced unused outputs won't
    receive gradients, causing DDP to hang or error.

    This function adds `0.0 * value.sum()` terms for all unused tensor outputs, which:
    - Forces gradients to flow through all parameters
    - Adds zero to the loss (no effect on optimization)
    - Allows DDP to synchronize all gradients

    **Performance Note (benchmarked December 2024):**

    We compared this approach vs PyTorch's `find_unused_parameters=True` DDP option:

    - **GPU (4x NVIDIA L4, nccl)**: This hack is ~7% faster (4.2ms vs 4.5ms per iter)
    - **CPU (gloo)**: `find_unused_parameters=True` is ~6% faster (6.6ms vs 7.0ms per iter)

    The difference is because:
    - `find_unused_parameters=True` traverses the autograd graph every iteration (~0.1ms CPU overhead)
    - On GPU, compute is fast so this overhead is relatively larger (~2.5% of iteration)
    - On CPU, compute is slow so the overhead is smaller (~1.5% of iteration)
    - `.sum()` is highly optimized on GPU (parallel CUDA kernel)

    Since most training is GPU-based, we use this approach by default.

    See `metta/rl/ddp_unused_params/benchmark.py` to reproduce these benchmarks.

    """
    for key in td.keys():
        if key not in used_keys and isinstance(td[key], Tensor):
            value = td[key]
            if value.requires_grad:
                loss = loss + 0.0 * value.sum()
    return loss
