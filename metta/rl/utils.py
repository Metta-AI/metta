"""Training utilities for Metta RL."""

from typing import Collection, Tuple

import torch
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite

_POLICY_METADATA_CACHE: dict[tuple[str, int, int], tuple[Tensor, Tensor]] = {}


def _get_policy_metadata_tensors(
    device: torch.device,
    batch_size: int,
    time_steps: int,
) -> tuple[Tensor, Tensor]:
    key = (str(device), batch_size, time_steps)
    cached = _POLICY_METADATA_CACHE.get(key)
    if cached is None or cached[0].device != device:
        total = batch_size * time_steps
        batch_tensor = torch.full((total,), batch_size, dtype=torch.long, device=device)
        bptt_tensor = torch.full((total,), time_steps, dtype=torch.long, device=device)
        _POLICY_METADATA_CACHE[key] = (batch_tensor, bptt_tensor)
        return batch_tensor, bptt_tensor
    return cached


def ensure_sequence_metadata(td: TensorDict, *, batch_size: int, time_steps: int) -> None:
    """Attach required sequence metadata to ``td`` if missing."""

    device = td.device
    batch_tensor, bptt_tensor = _get_policy_metadata_tensors(device, batch_size, time_steps)
    if "batch" not in td.keys():
        td.set("batch", batch_tensor)
    if "bptt" not in td.keys():
        td.set("bptt", bptt_tensor)


def prepare_policy_forward_td(
    minibatch: TensorDict,
    spec: Composite,
    *,
    clone: bool = True,
) -> Tuple[TensorDict, int, int]:
    """Prepare a TensorDict for policy forward pass with BPTT and batch metadata.

    This function extracts the relevant keys from a minibatch, optionally clones them,
    reshapes to a flat batch dimension, and sets the bptt and batch metadata required
    for policy forward passes.
    """
    td = minibatch.select(*spec.keys(include_nested=True))
    if clone:
        td = td.clone()

    B, TT = td.batch_size
    td = td.reshape(B * TT)
    batch_tensor, bptt_tensor = _get_policy_metadata_tensors(td.device, B, TT)
    td.set("bptt", bptt_tensor)
    td.set("batch", batch_tensor)

    return td, B, TT


def forward_policy_for_training(
    policy,
    minibatch: TensorDict,
    policy_spec: Composite,
) -> TensorDict:
    """Forward policy on sampled minibatch for training.

    Centralized forward pass for use in core training loop.
    """
    policy_td, B, TT = prepare_policy_forward_td(minibatch, policy_spec, clone=False)

    flat_actions = minibatch["actions"].reshape(B * TT, -1)

    policy.reset_memory()
    policy_td = policy.forward(policy_td, action=flat_actions)

    policy_td = policy_td.reshape(B, TT)

    return policy_td


def should_run(
    epoch: int,
    interval: int,
    *,
    force: bool = False,
) -> bool:
    """Check if a periodic task should run based on interval. It is assumed this is only called on master."""
    if not interval:
        return False

    if force:
        return True

    return epoch % interval == 0


def add_dummy_loss_for_unused_params(
    loss: Tensor,
    *,
    td: TensorDict,
    used_keys: Collection[str],
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

    Args:
        loss: The current loss tensor to augment.
        td: TensorDict containing policy outputs (some of which may be unused).
        used_keys: Keys in `td` that are already used in the loss computation.
            All other tensor keys with `requires_grad=True` will have dummy terms added.

    Returns:
        The loss tensor with dummy terms added for unused outputs.

    Example:
        >>> loss = -log_probs.mean()  # Only uses act_log_prob
        >>> loss = add_dummy_loss_for_unused_params(
        ...     loss, td=policy_td, used_keys=["act_log_prob", "entropy"]
        ... )
        >>> loss.backward()  # Now all params get gradients
    """
    for key in td.keys():
        if key not in used_keys and isinstance(td[key], Tensor):
            value = td[key]
            if value.requires_grad:
                loss = loss + 0.0 * value.sum()
    return loss
