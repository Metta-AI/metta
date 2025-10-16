"""Random policy implementation for CoGames."""

from typing import Any, Optional

from cogames.policy.interfaces import AgentPolicy, Policy
from mettagrid import MettaGridAction, MettaGridEnv, MettaGridObservation, dtype_actions


class RandomAgentPolicy(AgentPolicy):
    """Per-agent random policy."""

    def __init__(self, action_space):
        self._action_space = action_space

    def step(self, obs: MettaGridObservation) -> MettaGridAction:
        """Get random action.

        Args:
            obs: The observation (unused for random policy)

        Returns:
            A random action sampled from the action space
        """
        sample = self._action_space.sample()
        return dtype_actions.type(sample)


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

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        """Get an AgentPolicy instance for a specific agent.

        Args:
            agent_id: The ID of the agent

        Returns:
            A RandomAgentPolicy instance
        """
        return RandomAgentPolicy(self._action_space)
