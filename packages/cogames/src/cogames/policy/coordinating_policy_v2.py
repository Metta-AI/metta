"""Policy wrapper for CoordinatingAgentV2 (simple assembler coordination)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

from cogames.policy.interfaces import AgentPolicy, Policy, StatefulPolicyImpl
from cogames.policy.scripted_agent.coordinating_agent_v2 import CoordinatingAgentV2
from cogames.policy.scripted_agent.simple_baseline_agent import SimpleAgentState

if TYPE_CHECKING:
    from cogames.cogs_vs_clips.env import MettaGridEnv
    from cogames.cogs_vs_clips.observation import MettaGridObservation


class CoordinatingPolicyV2Impl(StatefulPolicyImpl[SimpleAgentState]):
    """Implementation that wraps CoordinatingAgentV2."""

    def __init__(self, env: MettaGridEnv):
        self._agent = CoordinatingAgentV2(env)
        self._env = env

    def agent_state(self, agent_id: int = 0) -> SimpleAgentState:
        """Get initial state for an agent."""
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
        agent_id = state.agent_id
        self._agent._agent_states[agent_id] = state
        action = self._agent.step(agent_id, obs)
        return action, self._agent._agent_states[agent_id]


class CoordinatingPolicyV2(Policy):
    """Policy class for coordinating agent V2 (simple assembler coordination)."""

    def __init__(self, env: MettaGridEnv | None = None, device=None):
        self._env = env
        self._device = device
        self._impl = CoordinatingPolicyV2Impl(env) if env is not None else None
        self._agent_policies: Dict[int, AgentPolicy] = {}

    def reset(self, obs, info):
        if self._env is None and "env" in info:
            self._env = info["env"]
        if self._impl is None:
            if self._env is None:
                raise RuntimeError("CoordinatingPolicyV2 needs env")
            self._impl = CoordinatingPolicyV2Impl(self._env)

        # Clear cached agent policies on reset
        self._agent_policies.clear()

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        if self._impl is None:
            raise RuntimeError("Policy not initialized - call reset() first")
        if agent_id not in self._agent_policies:
            from cogames.policy.interfaces import StatefulAgentPolicy

            self._agent_policies[agent_id] = StatefulAgentPolicy(self._impl, agent_id)
        return self._agent_policies[agent_id]
