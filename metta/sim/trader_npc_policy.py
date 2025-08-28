"""Simple NPC trader policy that wanders randomly and maintains inventory for trading."""

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict


class TraderNPCPolicy(nn.Module):
    """
    Simple NPC policy for trader agents that:
    1. Wanders randomly around the map
    2. Maintains inventory (batteries) for trading
    3. Never attacks or uses other complex actions
    """

    def __init__(
        self,
        num_actions: int = 9,  # Standard action space size
        wander_prob: float = 0.7,  # Probability of moving vs rotating
        seed: int = 42,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.wander_prob = wander_prob
        self.rng = np.random.RandomState(seed)

        # Action indices (standard MettaGrid action space)
        self.NOOP = 0
        self.MOVE = 1
        self.ROTATE = 2

        # Dummy parameters to make it compatible with policy loading
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, obs: TensorDict) -> TensorDict:
        """Generate random wandering actions for trader NPCs."""
        batch_size = obs["grid_obs"].shape[0] if "grid_obs" in obs else obs.shape[0]

        # Generate random actions: mostly move and rotate
        actions = torch.zeros((batch_size, 2), dtype=torch.uint8)

        for i in range(batch_size):
            if self.rng.random() < self.wander_prob:
                # Move forward
                actions[i, 0] = self.MOVE
                actions[i, 1] = 0  # No argument for move
            else:
                # Rotate randomly
                actions[i, 0] = self.ROTATE
                actions[i, 1] = self.rng.randint(0, 4)  # Random rotation direction

        # Return in expected format
        return TensorDict(
            {
                "actions": actions,
                "action": actions[:, 0],  # Primary action
                "action_arg": actions[:, 1],  # Action argument
            },
            batch_size=[batch_size],
        )

    def get_actions(self, obs: torch.Tensor) -> torch.Tensor:
        """Alternative interface for getting actions."""
        td = TensorDict({"grid_obs": obs}, batch_size=[obs.shape[0]])
        result = self.forward(td)
        return result["actions"]


def create_trader_npc_policy(**kwargs) -> TraderNPCPolicy:
    """Factory function to create trader NPC policy."""
    return TraderNPCPolicy(**kwargs)
