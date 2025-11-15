"""Training utilities for Metta RL."""

from typing import Tuple

import torch
from tensordict import TensorDict
from torchrl.data import Composite


def ensure_sequence_metadata(td: TensorDict, *, batch_size: int, time_steps: int) -> None:
    """Attach required sequence metadata to ``td`` if missing."""

    total = batch_size * time_steps
    device = td.device
    if "batch" not in td.keys():
        td.set("batch", torch.full((total,), batch_size, dtype=torch.long, device=device))
    if "bptt" not in td.keys():
        td.set("bptt", torch.full((total,), time_steps, dtype=torch.long, device=device))


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
    td.set("bptt", torch.full((B * TT,), TT, device=td.device, dtype=torch.long))
    td.set("batch", torch.full((B * TT,), B, device=td.device, dtype=torch.long))

    return td, B, TT


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
