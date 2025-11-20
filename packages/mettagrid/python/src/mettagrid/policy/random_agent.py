"""Random policy implementation for CoGames."""

import random

from mettagrid.policy.policy import AgentStepMixin, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface


class RandomMultiAgentPolicy(AgentStepMixin, MultiAgentPolicy):
    """Random multi-agent policy that samples actions uniformly from the action space."""

    short_names = ["random"]

    def __init__(self, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info)

    def agent_step(self, agent_id: int, obs):
        return random.choice(self.policy_env_info.actions.actions())

    def agent_policies(self, num_agents: int):
        return [self.agent_policy(i) for i in range(num_agents)]
