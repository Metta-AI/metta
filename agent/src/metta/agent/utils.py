"""Utility functions for agent operations."""

import numpy as np
import torch
from tensordict import TensorDict


def obs_to_td(obs: np.ndarray, device: str | torch.device = "cpu") -> TensorDict:
    """Convert numpy observations to TensorDict with 'env_obs' key."""
    return TensorDict({"env_obs": torch.from_numpy(obs).to(device)}, batch_size=obs.shape[0])
