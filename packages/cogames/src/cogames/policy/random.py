"""Random policy implementation for CoGames."""

from __future__ import annotations

from typing import Optional

import numpy as np

from cogames.policy.policy import AgentPolicy, Policy
from mettagrid import MettaGridAction, MettaGridEnv, MettaGridObservation


class RandomPolicy(Policy, AgentPolicy):
    """Random policy that samples actions uniformly from the action space."""

    def __init__(self, env: MettaGridEnv, device: Optional[object] = None):
        """Initialize random policy.

        Args:
            env: The environment to sample actions from
            device: Device to use (ignored for random policy)
        """
        self._env = env
        self._action_space = env.single_action_space

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        return self

    def step(self, obs: MettaGridObservation) -> MettaGridAction:
        """Get random action."""
        sample = self._action_space.sample()
        return np.asarray(sample, dtype=np.int32)

    def reset(self) -> None:
        """Reset the policy state (no-op for random policy)."""
        pass
