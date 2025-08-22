"""Utility functions for agent operations."""

import numpy as np
import torch
from tensordict import TensorDict


def obs_to_td(obs: np.ndarray, device: str | torch.device = "cpu") -> TensorDict:
    """Convert numpy observations to TensorDict.

    Args:
        obs: Numpy array of observations. Accepts shapes like
             [batch, tokens, 3] or [*, batch_like, tokens, 3].
        device: Device to place tensors on

    Returns:
        TensorDict with "env_obs" key. Leading batch-like dims are flattened so
        the resulting shape is [B_TT, tokens, 3].
    """
    # Ensure numpy array
    obs = np.asarray(obs)

    # Flatten any leading batch/time/env dims into a single batch dimension
    if obs.ndim >= 3:
        obs_flat = obs.reshape(-1, obs.shape[-2], obs.shape[-1])
    else:
        # Fallback: if unexpected rank, add a batch dimension
        obs_flat = obs.reshape(1, *obs.shape)

    env_obs = torch.from_numpy(obs_flat).to(device)
    return TensorDict({"env_obs": env_obs}, batch_size=env_obs.shape[0])
