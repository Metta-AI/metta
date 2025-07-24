"""Checkpoint NPC inference utilities for dual-policy training."""

import logging
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from metta.rl.experience import Experience
from metta.rl.util.rollout import get_lstm_config, run_policy_inference

logger = logging.getLogger(__name__)


class SimpleSpace:
    """Simple space-like object for observation and action spaces."""

    def __init__(self, shape, dtype):
        self.shape = shape
        # Convert PyTorch dtype to NumPy dtype for compatibility
        if hasattr(dtype, "numpy"):
            self.dtype = dtype.numpy()
        elif dtype == torch.int32:
            self.dtype = np.int32
        elif dtype == torch.int64:
            self.dtype = np.int64
        elif dtype == torch.float32:
            self.dtype = np.float32
        elif dtype == torch.float64:
            self.dtype = np.float64
        else:
            self.dtype = np.float32  # Default fallback


def run_checkpoint_npc_inference(
    npc_policy: torch.nn.Module,
    observations: Tensor,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor, Optional[dict]]:
    """Run inference for checkpoint-based NPC policies.

    Args:
        npc_policy: The loaded NPC policy
        observations: Input observations
        device: Device to run inference on

    Returns:
        Tuple of (actions, log_probs, values, lstm_state)
    """
    npc_batch_size = observations.shape[0]
    obs_shape = observations.shape[1:]  # Remove batch dimension
    atn_shape = (2,)  # Action shape is (action_type, action_param)

    # Create space objects
    obs_space = SimpleSpace(obs_shape, observations.dtype)
    atn_space = SimpleSpace(atn_shape, np.int32)

    # Get LSTM configuration from the NPC policy
    hidden_size, num_lstm_layers = get_lstm_config(npc_policy)

    # Create experience object
    npc_experience = Experience(
        total_agents=npc_batch_size,
        batch_size=npc_batch_size,
        bptt_horizon=1,
        minibatch_size=npc_batch_size,
        max_minibatch_size=npc_batch_size,
        obs_space=obs_space,
        atn_space=atn_space,
        device=device,
        hidden_size=hidden_size,
        num_lstm_layers=num_lstm_layers,
    )

    # Run inference
    return run_policy_inference(npc_policy, observations, npc_experience, 0, device)
