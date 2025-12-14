"""Random policy implementation for CoGames."""

import random
from typing import Optional

from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy, PolicyDescriptor
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action, AgentObservation


class RandomAgentPolicy(AgentPolicy):
    """Per-agent random policy."""

    def __init__(self, policy_env_info: PolicyEnvInterface, policy_descriptor: PolicyDescriptor):
        super().__init__(policy_env_info, policy_descriptor)

    def step(self, obs: AgentObservation) -> Action:
        return random.choice(self.policy_env_info.actions.actions())

    def reset(self) -> None:
        """Random policy keeps no state."""
        pass


class RandomMultiAgentPolicy(MultiAgentPolicy):
    """Random multi-agent policy that samples actions uniformly from the action space."""

    short_names = ["random"]

    def __init__(self, policy_env_info: PolicyEnvInterface,  policy_name: Optional[str] = None):
        super().__init__(policy_env_info, policy_name=policy_name)

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        """Get an AgentPolicy instance for a specific agent.

        Args:
            agent_id: The ID of the agent

        Returns:
            A RandomAgentPolicy instance
        """
        return RandomAgentPolicy(self._policy_env_info, self._policy_descriptor)

    def agent_policies(self, num_agents: int) -> list[AgentPolicy]:
        """Get a list of AgentPolicy instances for all agents."""
        return [self.agent_policy(i) for i in range(num_agents)]
