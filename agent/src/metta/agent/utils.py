"""Utility functions for agent operations."""

import numpy as np
import torch
from tensordict import TensorDict


def obs_to_td(obs: np.ndarray, device: str | torch.device = "cpu") -> TensorDict:
    """Convert numpy observations to TensorDict with standard metadata for policy inference."""

    env_obs = torch.from_numpy(obs).to(device)
    batch_size = env_obs.shape[0]
    td = TensorDict({"env_obs": env_obs}, batch_size=(batch_size,))
    td.set("bptt", torch.ones(batch_size, dtype=torch.long, device=env_obs.device))
    td.set("batch", torch.full((batch_size,), batch_size, dtype=torch.long, device=env_obs.device))
    return td
