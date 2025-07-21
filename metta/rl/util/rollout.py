"""Rollout phase functions for Metta training."""

import logging
from typing import Any, Tuple

import torch
from tensordict import TensorDict
from torch import Tensor

logger = logging.getLogger(__name__)


def get_observation(
    vecenv: Any,
    device: torch.device,
    timer: Any,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, list, slice, Tensor, int]:
    """Get observations and other data from the vectorized environment and convert to tensors.

    Returns:
        Tuple of (observations, rewards, dones, truncations, info, training_env_id, mask, num_steps)
    """
    # Receive environment data
    with timer("_rollout.env"):
        o, r, d, t, info, env_id, mask = vecenv.recv()

    training_env_id = slice(env_id[0], env_id[-1] + 1)

    mask = torch.as_tensor(mask)
    num_steps = int(mask.sum().item())

    # Convert to tensors
    o = torch.as_tensor(o).to(device, non_blocking=True)
    r = torch.as_tensor(r).to(device, non_blocking=True)
    d = torch.as_tensor(d).to(device, non_blocking=True)
    t = torch.as_tensor(t).to(device, non_blocking=True)

    return o, r, d, t, info, training_env_id, mask, num_steps


def run_policy_inference(
    policy: torch.nn.Module,
    input_td: TensorDict,
    device: torch.device,
) -> TensorDict:
    """Run the policy to get actions and value estimates.

    Returns:
        A TensorDict containing all data to be stored in the experience buffer.
    """
    with torch.no_grad():
        experience_td = policy(input_td)

        if str(device).startswith("cuda"):
            torch.cuda.synchronize()

    return experience_td
