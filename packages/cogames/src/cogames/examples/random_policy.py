"""Random policy implementation for CoGames."""

from typing import Any, Optional

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

    def step(self, agent_id: int, agent_obs: Any) -> Any:
        """Get random action for a single agent.

        Args:
            agent_id: The ID of the agent (unused for random policy)
            agent_obs: The observation for this specific agent (unused for random policy)

        Returns:
            A random action sampled from the action space
        """
        # Return a single random action for this agent
        return self.action_space.sample()

    def reset(self) -> None:
        """Reset the policy state (no-op for random policy)."""
        pass
