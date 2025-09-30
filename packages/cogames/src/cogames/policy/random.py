"""Random policy implementation for CoGames."""

from typing import Optional

import numpy as np

from cogames.policy.policy import AgentPolicy, Policy
from mettagrid import MettaGridAction, MettaGridEnv, MettaGridObservation


class RandomPolicy(Policy):
    """Creates independent per-agent random policies."""

    class _Agent(AgentPolicy):
        def __init__(self, action_space):
            self._action_space = action_space

        def step(self, obs: MettaGridObservation) -> MettaGridAction:
            sample = self._action_space.sample()
            return np.asarray(sample, dtype=np.int32)

        def reset(self) -> None:
            return None

    def __init__(self, env: MettaGridEnv, device: Optional[object] = None):
        self._env = env
        self._action_space = env.single_action_space

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        return self._Agent(self._action_space)
