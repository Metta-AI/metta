"""Utility functions for agent operations."""

import numpy as np
import torch
from tensordict import TensorDict


def obs_to_td(obs: np.ndarray, device: str | torch.device = "cpu") -> TensorDict:
    """Convert numpy observations to TensorDict with 'env_obs' key.

    Handles format standardization between different environment types:
    - Direct MettaGridEnv (single agent): [tokens, features] -> [1, tokens, features]
    - Vectorized environments: [agents, tokens, features] -> [agents, tokens, features]

    This ensures consistent tensor dimensions for downstream policy networks.
    """
    obs_tensor = torch.from_numpy(obs).to(device)

    # Standardize observation format: ensure we always have [batch, tokens, features]
    if obs_tensor.ndim == 2:
        # Single agent direct environment format: [tokens, features]
        # Add batch dimension: [tokens, features] -> [1, tokens, features]
        obs_tensor = obs_tensor[None, ...]
        batch_size = 1
    elif obs_tensor.ndim == 3:
        # Multi-agent/vectorized environment format: [agents, tokens, features]
        batch_size = obs.shape[0]
    else:
        raise ValueError(
            f"Unexpected observation dimensionality: {obs_tensor.ndim}D. "
            f"Expected 2D [tokens, features] or 3D [agents, tokens, features], got shape {obs.shape}"
        )

    return TensorDict({"env_obs": obs_tensor}, batch_size=batch_size)
