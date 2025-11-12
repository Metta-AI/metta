"""Noop policy implementation."""

import numpy as np

from mettagrid.mettagrid_c import dtype_actions
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action, AgentObservation


class NoopAgentPolicy(AgentPolicy):
    """Per-agent noop policy."""

    def __init__(self, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info)
        self._noop_index = policy_env_info.action_names.index("noop")

    def step(self, obs: AgentObservation) -> Action:
        """Return the noop action for the agent."""
        return self._policy_env_info.actions.noop.Noop()

    def step_batch(self, raw_observations: np.ndarray, raw_actions: np.ndarray) -> None:
        """Fill the provided raw action buffer with noop actions."""
        noop_value = dtype_actions.type(self._noop_index)
        raw_actions[...] = noop_value


class NoopPolicy(MultiAgentPolicy):
    """Policy that always selects the noop action when available."""

    short_names = ["noop"]

    def __init__(self, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info)

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        """Get an AgentPolicy instance configured with the noop action id."""
        return NoopAgentPolicy(self._policy_env_info)
