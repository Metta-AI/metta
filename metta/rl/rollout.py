"""Rollout phase functions for Metta training."""

import logging
import typing

import numpy as np
import torch

import mettagrid.profiling.stopwatch

logger = logging.getLogger(__name__)


PufferlibVecEnv = typing.Any


def get_observation(
    vecenv: PufferlibVecEnv,
    device: torch.device,
    timer: mettagrid.profiling.stopwatch.Stopwatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list, slice, torch.Tensor, int]:
    """Get observations from vectorized environment and convert to tensors."""
    with timer("_rollout.env"):
        o, r, d, t, ta, info, env_id, mask = vecenv.recv()

    training_env_id = slice(env_id[0], env_id[-1] + 1)

    mask = torch.as_tensor(mask)
    num_steps = int(mask.sum().item())

    # Convert to tensors
    o = torch.as_tensor(o).to(device, non_blocking=True)
    r = torch.as_tensor(r).to(device, non_blocking=True)
    d = torch.as_tensor(d).to(device, non_blocking=True)
    t = torch.as_tensor(t).to(device, non_blocking=True)

    return o, r, d, t, info, training_env_id, mask, num_steps


def send_observation(
    vecenv: PufferlibVecEnv,
    actions: torch.Tensor,
    dtype_actions: np.dtype,
    timer: mettagrid.profiling.stopwatch.Stopwatch,
) -> None:
    """Send actions back to the vectorized environment."""
    with timer("_rollout.env"):
        vecenv.send(actions.cpu().numpy().astype(dtype_actions))
