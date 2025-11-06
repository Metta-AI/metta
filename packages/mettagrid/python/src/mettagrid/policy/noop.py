"""Noop policy implementation."""

import mettagrid.policy.policy
import mettagrid.policy.policy_env_interface
import mettagrid.simulator


class NoopAgentPolicy(mettagrid.policy.policy.AgentPolicy):
    """Per-agent noop policy."""

    def __init__(self, policy_env_info: mettagrid.policy.policy_env_interface.PolicyEnvInterface):
        super().__init__(policy_env_info)

    def step(self, obs: mettagrid.simulator.AgentObservation) -> mettagrid.simulator.Action:
        """Return the noop action for the agent."""
        return self._policy_env_info.actions.noop.Noop()


class NoopPolicy(mettagrid.policy.policy.MultiAgentPolicy):
    """Policy that always selects the noop action when available."""

    def __init__(self, policy_env_info: mettagrid.policy.policy_env_interface.PolicyEnvInterface):
        super().__init__(policy_env_info)

    def agent_policy(self, agent_id: int) -> mettagrid.policy.policy.AgentPolicy:
        """Get an AgentPolicy instance configured with the noop action id."""
        return NoopAgentPolicy(self._policy_env_info)
