"""Random policy implementation for CoGames."""

from typing import Any, Optional

import numpy as np

from cogames.policy import Policy
from mettagrid import MettaGridEnv


class RandomPolicy(Policy):
    """Random policy that samples actions uniformly from the action space."""

    def __init__(self, env: MettaGridEnv, device: Optional[Any] = None):
        """Initialize random policy.

        Args:
            env: The environment to sample actions from
            device: Device to use (ignored for random policy)
        """
        self.env = env
        self.action_space = env.single_action_space

    def step(self, observations: list[Any]) -> np.ndarray:
        """Get random actions for given observations.

        Args:
            observations: The current observations from the environment

        Returns:
            Random actions sampled from the action space
        """
        # Return a list of actions, one for each observation
        return np.array([self.action_space.sample() for _ in observations])

    def reset(self) -> None:
        """Reset the policy state (no-op for random policy)."""
        pass
