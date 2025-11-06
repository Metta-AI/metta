"""Utility functions for agent operations."""

import numpy as np
import tensordict
import torch

import metta.rl.utils


def obs_to_td(obs: np.ndarray, device: str | torch.device = "cpu") -> tensordict.TensorDict:
    """Convert numpy observations to TensorDict with standard metadata for policy inference."""

    env_obs = torch.from_numpy(obs).to(device)
    batch_size = env_obs.shape[0]
    td = tensordict.TensorDict({"env_obs": env_obs}, batch_size=(batch_size,))
    metta.rl.utils.ensure_sequence_metadata(td, batch_size=batch_size, time_steps=1)
    td.set("env_obs_flat", env_obs.view(batch_size, -1))
    return td
