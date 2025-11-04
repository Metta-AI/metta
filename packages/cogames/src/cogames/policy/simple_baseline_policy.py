"""Policy wrapper for SimpleBaselineAgent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

from cogames.policy.interfaces import AgentPolicy, Policy, StatefulPolicyImpl
from cogames.policy.scripted_agent.simple_baseline_agent import SimpleAgentState, SimpleBaselineAgent

if TYPE_CHECKING:
    from cogames.cogs_vs_clips.env import MettaGridEnv
    from cogames.cogs_vs_clips.observation import MettaGridObservation


class SimpleBaselinePolicyImpl(StatefulPolicyImpl[SimpleAgentState]):
    """Implementation that wraps SimpleBaselineAgent."""

    def __init__(self, env: MettaGridEnv):
        self._agent = SimpleBaselineAgent(env)
        self._env = env

    def agent_state(self, agent_id: int = 0) -> SimpleAgentState:
        """Get initial state for an agent."""
        # Make sure agent states are initialized
        if agent_id not in self._agent._agent_states:
            from cogames.policy.scripted_agent.simple_baseline_agent import CellType

            self._agent._agent_states[agent_id] = SimpleAgentState(
                agent_id=agent_id,
                map_height=self._agent._map_h,
                map_width=self._agent._map_w,
                occupancy=[[CellType.FREE.value] * self._agent._map_w for _ in range(self._agent._map_h)],
            )
        return self._agent._agent_states[agent_id]

    def step_with_state(self, obs: MettaGridObservation, state: SimpleAgentState) -> tuple[int, SimpleAgentState]:
        """Compute action and return updated state."""
        # The state passed in tells us which agent this is
        agent_id = state.agent_id
        # Update the shared agent state
        self._agent._agent_states[agent_id] = state
        # Compute action
        action = self._agent.step(agent_id, obs)
        # Return action and updated state
        return action, self._agent._agent_states[agent_id]


class SimpleBaselinePolicy(Policy):
    """Policy class for simple baseline agent (matches ScriptedAgentPolicy pattern)."""

    def __init__(self, env: MettaGridEnv | None = None, device=None):
        self._env = env
        self._impl = SimpleBaselinePolicyImpl(env) if env is not None else None
        self._agent_policies: Dict[int, AgentPolicy] = {}  # Cache per-agent policies

    def reset(self, obs, info):
        if self._env is None and "env" in info:
            self._env = info["env"]
        if self._impl is None:
            if self._env is None:
                raise RuntimeError("SimpleBaselinePolicy needs env - provide during __init__ or via info['env']")
            self._impl = SimpleBaselinePolicyImpl(self._env)

        # Extract agent spawn positions from info if available (for tests)
        if "agent_spawn_positions" in info and self._impl is not None:
            self._impl._agent._agent_spawn_positions = info["agent_spawn_positions"]

        # Clear cached agent policies on reset
        self._agent_policies.clear()

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        if self._impl is None:
            raise RuntimeError("Policy not initialized - call reset() first")
        # Create agent policies lazily
        if agent_id not in self._agent_policies:
            from cogames.policy.interfaces import StatefulAgentPolicy

            self._agent_policies[agent_id] = StatefulAgentPolicy(self._impl, agent_id)
        return self._agent_policies[agent_id]
