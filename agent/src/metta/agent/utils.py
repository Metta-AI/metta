"""Utility functions for agent operations."""

import numpy as np
import torch
from tensordict import TensorDict

from metta.rl.utils import ensure_sequence_metadata


def obs_to_td(obs: np.ndarray, device: str | torch.device = "cpu") -> TensorDict:
    """Convert numpy observations to TensorDict with standard metadata for policy inference."""

    env_obs = torch.from_numpy(obs).to(device)
    batch_size = env_obs.shape[0]
    td = TensorDict({"env_obs": env_obs}, batch_size=(batch_size,))
    ensure_sequence_metadata(td, batch_size=batch_size, time_steps=1)
    return td


def resolve_torch_dtype(dtype_str: str | None) -> torch.dtype:
    """Resolve common dtype strings to torch dtypes with float32 default."""
    if dtype_str is None:
        return torch.float32
    s = str(dtype_str).lower()
    mapping = {"float16": torch.float16, "fp16": torch.float16, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}
    return mapping.get(s, torch.float32)
