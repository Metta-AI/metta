"""Random policy implementation for CoGames."""

from typing import Optional

import numpy as np

from cogames.policy.policy import AgentPolicy, Policy
from mettagrid import MettaGridAction, MettaGridEnv, MettaGridObservation


class RandomPolicy(Policy, AgentPolicy):
    """Random policy that samples actions uniformly from the action space."""

    def __init__(self, env: MettaGridEnv, device: Optional[object] = None):
        """Create a random policy that samples uniformly from the action space."""
        self._env = env
        self._action_space = env.single_action_space

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        return self

    def step(self, obs: MettaGridObservation) -> MettaGridAction:
        """Sample a random action ignoring the observation."""
        sample = self._action_space.sample()
        return np.asarray(sample, dtype=np.int32)

    def reset(self) -> None:
        """Reset policy state (no-op)."""
        pass
