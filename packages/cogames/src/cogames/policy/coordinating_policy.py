"""Policy wrapper for CoordinatingAgent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

from cogames.policy.scripted_agent.coordinating_agent import CoordinatingAgent, CoordinatingAgentState

if TYPE_CHECKING:
    from mettagrid import MettaGridEnv, MettaGridObservation


class AgentPolicy:
    """Per-agent policy wrapper."""

    def __init__(self, impl: CoordinatingAgent, agent_id: int):
        self._impl = impl
        self._agent_id = agent_id

    def step(self, obs: MettaGridObservation) -> int:
        """Compute action for this agent."""
        return self._impl.step(self._agent_id, obs)


class CoordinatingPolicy:
    """Policy wrapper for CoordinatingAgent with per-agent views."""

    def __init__(self, env: MettaGridEnv | None = None, device=None):
        self._env = env
        self._device = device  # Not used for scripted agents but needed for interface
        self._impl = CoordinatingAgent(env) if env is not None else None
        self._agent_policies: Dict[int, AgentPolicy] = {}

    def reset(self, obs, info):
        """Reset policy state."""
        # Get environment from info if not provided at init
        if self._env is None:
            self._env = info.get("env")

        # Initialize implementation if needed
        if self._impl is None:
            if self._env is None:
                raise RuntimeError("CoordinatingPolicy needs env - provide during __init__ or via info['env']")
            self._impl = CoordinatingAgent(self._env)

        # Reset agent states and shared state
        self._impl._agent_states.clear()
        self._agent_policies.clear()

        # Reset shared coordination state
        CoordinatingAgent._assembly_signal = {"active": False, "requester": None, "position": None}
        CoordinatingAgent._assembly_signal_participants = set()
        CoordinatingAgent._target_assignments = {}
        CoordinatingAgent._target_position_counts = {}
        CoordinatingAgent._reserved_adjacent_cells = {}

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        """Get policy for a specific agent."""
        if agent_id not in self._agent_policies:
            if self._impl is None:
                raise RuntimeError("Policy not initialized - call reset() first")

            # Initialize agent state if needed
            if agent_id not in self._impl._agent_states:
                from cogames.policy.scripted_agent.simple_baseline_agent import CellType

                self._impl._agent_states[agent_id] = CoordinatingAgentState(
                    agent_id=agent_id,
                    map_height=self._impl._map_h,
                    map_width=self._impl._map_w,
                    occupancy=[[CellType.FREE.value] * self._impl._map_w for _ in range(self._impl._map_h)],
                )

            self._agent_policies[agent_id] = AgentPolicy(self._impl, agent_id)

        return self._agent_policies[agent_id]

    def agent_state(self, agent_id: int = 0) -> CoordinatingAgentState:
        """Get state for an agent (for debugging/inspection)."""
        if agent_id not in self._impl._agent_states:
            from cogames.policy.scripted_agent.simple_baseline_agent import CellType

            self._impl._agent_states[agent_id] = CoordinatingAgentState(
                agent_id=agent_id,
                map_height=self._impl._map_h,
                map_width=self._impl._map_w,
                occupancy=[[CellType.FREE.value] * self._impl._map_w for _ in range(self._impl._map_h)],
            )
        return self._impl._agent_states[agent_id]
