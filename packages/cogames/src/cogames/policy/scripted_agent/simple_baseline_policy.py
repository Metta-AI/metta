"""Policy wrapper for SimpleBaselineAgentImpl."""

from __future__ import annotations

from cogames.policy.scripted_agent.simple_baseline_agent import SimpleBaselineAgentImpl
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy, StatefulAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface


class SimpleBaselinePolicy(MultiAgentPolicy):
    """Policy class for simple baseline agent."""

    def __init__(self, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info)
        self._policy_env_info = policy_env_info
        self._agent_policies: list[AgentPolicy] = [
            StatefulAgentPolicy(SimpleBaselineAgentImpl(self._policy_env_info, agent_id, simulation=None), agent_id)
            for agent_id in range(self._policy_env_info.num_agents)
        ]

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        """Get policy for a specific agent."""
        return self._agent_policies[agent_id]
