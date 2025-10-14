"""Noop policy implementation for CoGames."""

from typing import Any, Optional

from cogames.policy.interfaces import AgentPolicy, Policy
from mettagrid import MettaGridAction, MettaGridEnv, MettaGridObservation, dtype_actions


class NoopAgentPolicy(AgentPolicy):
    """Per-agent noop policy."""

    def __init__(self, noop_action_id: int) -> None:
        self._noop_action_id = noop_action_id

    def step(self, obs: MettaGridObservation) -> MettaGridAction:
        """Return the noop action for the agent."""
        return dtype_actions.type(self._noop_action_id)


class NoopPolicy(Policy):
    """Policy that always selects the noop action when available."""

    def __init__(self, env: MettaGridEnv, device: Optional[Any] = None) -> None:
        self._env = env
        self._device = device
        self._noop_action_id = self._resolve_noop_action_id(env)

    @staticmethod
    def _resolve_noop_action_id(env: MettaGridEnv) -> int:
        """Return the noop index from the environment action names, defaulting to zero."""
        action_lookup = {name: idx for idx, name in enumerate(env.action_names)}
        return action_lookup.get("noop", 0)

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        """Get an AgentPolicy instance configured with the noop action id."""
        return NoopAgentPolicy(self._noop_action_id)
