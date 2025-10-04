"""Random policy implementation for CoGames."""

from typing import Optional

import numpy as np

from cogames.policy.policy import AgentPolicy, Policy
from mettagrid import MettaGridAction, MettaGridEnv, MettaGridObservation


class RandomAgentPolicy(AgentPolicy):
    """Per-agent random policy for hierarchical action spaces."""

    def __init__(self, action_space) -> None:
        self._action_space = action_space

    def step(self, obs: MettaGridObservation) -> MettaGridAction:
        """Get a random action sampled from the environment's action space."""
        sample = self._action_space.sample()
        return np.asarray(sample, dtype=np.int32)

    def reset(self) -> None:  # pragma: no cover - nothing to reset
        return None


class RandomPolicy(Policy):
    """Creates independent per-agent random policies."""

    def __init__(self, env: MettaGridEnv, device: Optional[object] = None) -> None:
        self._env = env
        self._action_space = env.single_action_space

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        return RandomAgentPolicy(self._action_space)
