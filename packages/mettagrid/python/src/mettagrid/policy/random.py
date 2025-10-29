"""Random policy implementation for CoGames."""

import random

from mettagrid.config.mettagrid_config import ActionsConfig
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action, AgentObservation


class RandomAgentPolicy(AgentPolicy):
    """Per-agent random policy."""

    def __init__(self, actions: ActionsConfig):
        super().__init__(actions)

    def step(self, obs: AgentObservation) -> Action:
        return random.choice(self._actions.actions())

    def reset(self) -> None:
        pass


class RandomMultiAgentPolicy(MultiAgentPolicy):
    """Random multi-agent policy that samples actions uniformly from the action space."""

    def __init__(self, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info)

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        """Get an AgentPolicy instance for a specific agent.

        Args:
            agent_id: The ID of the agent

        Returns:
            A RandomAgentPolicy instance
        """
        return RandomAgentPolicy(self._actions)

    def agent_policies(self, num_agents: int) -> list[AgentPolicy]:
        """Get a list of AgentPolicy instances for all agents."""
        return [self.agent_policy(i) for i in range(num_agents)]
