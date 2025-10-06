"""Random policy implementation for CoGames."""

from typing import Optional

import numpy as np

from cogames.policy.policy import AgentPolicy, Policy
from cogames.policy.utils import ActionLayout
from mettagrid import MettaGridAction, MettaGridEnv, MettaGridObservation


class RandomAgentPolicy(AgentPolicy):
    """Per-agent random policy that respects verb-specific argument limits."""

    def __init__(self, layout: ActionLayout) -> None:
        self._layout = layout

    def step(self, obs: MettaGridObservation) -> MettaGridAction:
        verb = np.random.randint(0, len(self._layout.max_args))
        arg = np.random.randint(0, int(self._layout.max_args[verb]) + 1)
        return np.asarray([verb, arg], dtype=np.int32)

    def reset(self) -> None:  # pragma: no cover - nothing to reset
        return None


class RandomPolicy(Policy):
    """Creates independent per-agent random policies."""

    def __init__(self, env: MettaGridEnv, device: Optional[object] = None) -> None:
        self._env = env
        self._layout = ActionLayout.from_env(env)

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        return RandomAgentPolicy(self._layout)
