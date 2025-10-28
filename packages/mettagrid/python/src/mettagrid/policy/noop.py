"""Noop policy implementation."""

from mettagrid.config.mettagrid_config import ActionsConfig
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.simulator import Action, AgentObservation


class NoopAgentPolicy(AgentPolicy):
    """Per-agent noop policy."""

    def __init__(self, actions: ActionsConfig):
        super().__init__(actions)

    def step(self, obs: AgentObservation) -> Action:
        """Return the noop action for the agent."""
        return self._actions.noop.Noop()


class NoopPolicy(MultiAgentPolicy):
    """Policy that always selects the noop action when available."""

    def __init__(self, actions: ActionsConfig):
        super().__init__(actions)

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        """Get an AgentPolicy instance configured with the noop action id."""
        return NoopAgentPolicy(self._actions)
