"""Random policy implementation for CoGames."""

import random

import mettagrid.policy.policy
import mettagrid.policy.policy_env_interface
import mettagrid.simulator


class RandomAgentPolicy(mettagrid.policy.policy.AgentPolicy):
    """Per-agent random policy."""

    def __init__(self, policy_env_info: mettagrid.policy.policy_env_interface.PolicyEnvInterface):
        super().__init__(policy_env_info)

    def step(self, obs: mettagrid.simulator.AgentObservation) -> mettagrid.simulator.Action:
        return random.choice(self._policy_env_info.actions.actions())

    def reset(self, simulation=None) -> None:
        pass


class RandomMultiAgentPolicy(mettagrid.policy.policy.MultiAgentPolicy):
    """Random multi-agent policy that samples actions uniformly from the action space."""

    def __init__(self, policy_env_info: mettagrid.policy.policy_env_interface.PolicyEnvInterface):
        super().__init__(policy_env_info)

    def agent_policy(self, agent_id: int) -> mettagrid.policy.policy.AgentPolicy:
        """Get an AgentPolicy instance for a specific agent.

        Args:
            agent_id: The ID of the agent

        Returns:
            A RandomAgentPolicy instance
        """
        return RandomAgentPolicy(self._policy_env_info)

    def agent_policies(self, num_agents: int) -> list[mettagrid.policy.policy.AgentPolicy]:
        """Get a list of AgentPolicy instances for all agents."""
        return [self.agent_policy(i) for i in range(num_agents)]
