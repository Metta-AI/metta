"""Random policy implementation for CoGames."""

from typing import Any, Optional

from cogames.policy.policy import Policy
from mettagrid import MettaGridEnv


class RandomPolicy(Policy):
    """Random policy that samples actions uniformly from the action space."""

    def __init__(self, env: MettaGridEnv, device: Optional[Any] = None):
        """Initialize random policy.

        Args:
            env: The environment to sample actions from
            device: Device to use (ignored for random policy)
        """
        self._env = env
        self._action_space = env.single_action_space

    def step(self, obs: Any) -> Any:
        """Get random action.

        Args:
            obs: The observation (unused for random policy)

        Returns:
            A random action sampled from the action space
        """
        # Return a random action
        return self._action_space.sample()

    def reset(self) -> None:
        """Reset the policy state (no-op for random policy)."""
        pass
