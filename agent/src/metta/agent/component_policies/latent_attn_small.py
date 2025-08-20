"""
LatentAttnSmall component policy implementation.
"""

import torch
import torch.nn as nn
from tensordict import TensorDict


class LatentAttnSmall(nn.Module):
    """LatentAttnSmall component policy implementation."""

    def __init__(self, obs_space=None, obs_width=None, obs_height=None, feature_normalizations=None, config=None):
        super().__init__()
        self.obs_space = obs_space
        self.obs_width = obs_width
        self.obs_height = obs_height
        self.feature_normalizations = feature_normalizations or {}
        self.config = config or {}

        # Simple network for demonstration
        input_size = obs_width * obs_height * 3 if obs_width and obs_height else 100
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 10),  # Default action space size
        )

    def forward(self, td: TensorDict, state=None, action=None) -> TensorDict:
        """Forward pass through the network."""
        obs = td["env_obs"]
        if len(obs.shape) == 4:  # (batch, height, width, channels)
            obs = obs.flatten(1)  # Flatten spatial dimensions

        logits = self.network(obs)
        actions = torch.argmax(logits, dim=-1)

        td["actions"] = actions
        return td

    def reset_memory(self):
        """Reset memory - not used in this simple agent."""
        pass

    def get_memory(self):
        """Get memory state - not used in this simple agent."""
        return {}
