"""Noop policy implementation."""

import numpy as np

from mettagrid.mettagrid_c import dtype_actions
from mettagrid.policy.policy import AgentStepMixin, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface


class NoopPolicy(AgentStepMixin, MultiAgentPolicy):
    """Policy that always selects the noop action when available."""

    short_names = ["noop"]

    def __init__(self, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info)
        self._noop_action_value = dtype_actions.type(policy_env_info.action_names.index("noop"))
        self._noop_action = policy_env_info.actions.noop.Noop()

    def agent_step(self, agent_id: int, obs):
        return self._noop_action

    def step_batch(self, raw_observations: np.ndarray, raw_actions: np.ndarray) -> None:
        raw_actions[...] = self._noop_action_value
