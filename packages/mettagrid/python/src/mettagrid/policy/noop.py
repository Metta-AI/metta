"""Noop policy implementation."""

from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action, AgentObservation


class NoopAgentPolicy(AgentPolicy):
    """Per-agent noop policy."""

    def __init__(self, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info)

    def step(self, obs: AgentObservation) -> Action:
        """Return the noop action for the agent."""
        return self._policy_env_info.actions.noop.Noop()


class NoopPolicy(MultiAgentPolicy):
    """Policy that always selects the noop action when available."""

    def __init__(self, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info)

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        """Get an AgentPolicy instance configured with the noop action id."""
        return NoopAgentPolicy(self._policy_env_info)
