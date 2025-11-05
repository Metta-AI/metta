"""
CoordinatingAgent - Multi-agent coordination for assemblers and extractors.

Extends UnclippingAgent with:
- Smart mouth selection for assemblers and extractors (agents spread around stations)
- Collision avoidance via random unsticking when blocked
- Commitment to chosen mouths to prevent oscillation

This agent has all capabilities:
- Core: resource gathering, assembling, delivery (from SimpleBaselineAgent)
- Unclipping: detect and restore clipped extractors (from UnclippingAgent)
- Coordination: multi-agent collision avoidance (this class)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

from .simple_baseline_agent import CellType
from .unclipping_agent import UnclippingAgent, UnclippingAgentState

if TYPE_CHECKING:
    from cogames.policy import AgentPolicy
    from mettagrid.simulator import Simulation


class CoordinatingAgent(UnclippingAgent):
    """
    Multi-agent coordination via smart mouth selection.

    Extends UnclippingAgent (which extends SimpleBaselineAgent), so has:
    - Core gathering/assembly/delivery from SimpleBaselineAgent
    - Unclipping capability from UnclippingAgent
    - Coordination via smart mouth selection (this class)

    When within 2 cells of assembler or extractor, picks an available adjacent
    cell (mouth) from observations to avoid collisions. Agents commit to their
    chosen mouth to prevent oscillation.
    """

    def __init__(self, simulation: "Simulation"):
        super().__init__(simulation)

    def _explore_frontier(self, s: UnclippingAgentState) -> int:
        """
        Override exploration with outward bias to spread agents out.

        Multi-agent scenarios benefit from agents exploring different areas,
        so we add a bonus for exploring away from the center (assembler/chest).
        """
        # Get the center point (assembler or chest if known)
        center = None
        if self._stations.get("assembler") is not None:
            center = self._stations["assembler"]
        elif self._stations.get("chest") is not None:
            center = self._stations["chest"]

        # If we don't know the center yet, use default exploration
        if center is None:
            return super()._explore_frontier(s)

        # Use parent's exploration logic but modify the scoring
        if s.row < 0:
            return self._NOOP

        # Initialize visited map if not exists
        if s.visited_map is None:
            s.visited_map = [[0 for _ in range(s.map_width)] for _ in range(s.map_height)]

        # Mark current cell as visited
        s.visited_map[s.row][s.col] = s.step_count

        import random

        from .simple_baseline_agent import CellType

        # 10% chance to take a random move to break patterns
        if random.random() < 0.1:
            valid_moves = []
            for action, (dr, dc) in [
                (self._MOVE_N, (-1, 0)),
                (self._MOVE_S, (1, 0)),
                (self._MOVE_E, (0, 1)),
                (self._MOVE_W, (0, -1)),
            ]:
                nr, nc = s.row + dr, s.col + dc
                if self._is_within_bounds(s, nr, nc) and s.occupancy[nr][nc] == CellType.FREE.value:
                    valid_moves.append(action)

            if valid_moves:
                s.exploration_target = None
                return random.choice(valid_moves)

        # Check if we should keep current exploration target
        if s.exploration_target is not None:
            tr, tc = s.exploration_target
            if s.step_count - s.exploration_target_step < 10 and (s.row, s.col) != s.exploration_target:
                if self._is_within_bounds(s, tr, tc) and s.occupancy[tr][tc] == CellType.FREE.value:
                    return self._move_towards(s, s.exploration_target)
            s.exploration_target = None

        # Find nearest least-recently-visited cell, WITH OUTWARD BIAS
        best_target = None
        best_score = -float("inf")
        center_r, center_c = center

        for radius in range(5, min(15, s.map_height // 2, s.map_width // 2)):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if abs(dr) != radius and abs(dc) != radius:
                        continue

                    r, c = s.row + dr, s.col + dc
                    if not self._is_within_bounds(s, r, c):
                        continue

                    if s.occupancy[r][c] != CellType.FREE.value:
                        continue

                    # Score based on visit time
                    last_visited = s.visited_map[r][c]
                    visit_score = s.step_count - last_visited
                    distance_to_me = abs(dr) + abs(dc)

                    # ADD OUTWARD BIAS: bonus for being far from center
                    distance_from_center = abs(r - center_r) + abs(c - center_c)
                    outward_bonus = distance_from_center * 5  # 5 points per cell away from center

                    # Combined score: prefer unvisited cells, close to me, far from center
                    score = visit_score * 10 - distance_to_me + outward_bonus

                    if score > best_score:
                        best_score = score
                        best_target = (r, c)

            if best_target and best_score > 100:
                break

        if best_target:
            s.exploration_target = best_target
            s.exploration_target_step = s.step_count
            return self._move_towards(s, best_target)

        # No good target found, pick a random direction
        valid_moves = []
        for action, (dr, dc) in [
            (self._MOVE_N, (-1, 0)),
            (self._MOVE_S, (1, 0)),
            (self._MOVE_E, (0, 1)),
            (self._MOVE_W, (0, -1)),
        ]:
            nr, nc = s.row + dr, s.col + dc
            if self._is_within_bounds(s, nr, nc) and s.occupancy[nr][nc] == CellType.FREE.value:
                valid_moves.append(action)

        if valid_moves:
            return random.choice(valid_moves)

        return self._NOOP

    # Note: Removed all mouth selection logic. CoordinatingAgent now simply:
    # 1. Uses outward exploration bias (via _explore_frontier override)
    # 2. Uses random unsticking every 10 steps (inherited from SimpleBaselineAgent)
    # This combination performs better than complex mouth coordination.


# ============================================================================
# Policy Wrapper Classes
# ============================================================================


class CoordinatingPolicyImpl:
    """Implementation that wraps CoordinatingAgent."""

    def __init__(self, simulation: "Simulation"):
        self._agent = CoordinatingAgent(simulation)
        self._sim = simulation

    def agent_state(self, agent_id: int = 0) -> UnclippingAgentState:
        """Get initial state for an agent."""
        # Make sure agent states are initialized
        if agent_id not in self._agent._agent_states:
            state = UnclippingAgentState(
                agent_id=agent_id,
                map_height=self._agent._map_h,
                map_width=self._agent._map_w,
                occupancy=[[CellType.FREE.value] * self._agent._map_w for _ in range(self._agent._map_h)],
            )
            # Initialize mutable defaults
            state.unreachable_extractors = {}
            state.agent_occupancy = set()
            self._agent._agent_states[agent_id] = state
        return self._agent._agent_states[agent_id]

    def step_with_state(self, obs, state: UnclippingAgentState):
        """Compute action and return updated state."""
        from mettagrid.simulator.interface import Action

        # The state passed in tells us which agent this is
        agent_id = state.agent_id
        # Update the shared agent state
        self._agent._agent_states[agent_id] = state
        # Compute action (returns integer index)
        action_idx = self._agent.step(agent_id, obs)
        # Convert to Action object
        action = Action(name=self._agent._action_names[action_idx])
        # Return action and updated state
        return action, self._agent._agent_states[agent_id]


class CoordinatingPolicy:
    """Policy class for coordinating agent.

    This policy requires a Simulation object for accessing grid_objects()
    to get absolute agent positions. Pass it via reset(simulation=sim).
    """

    def __init__(self):
        """Initialize policy (simulation will be provided via reset)."""
        self._sim = None
        self._impl = None
        self._agent_policies: Dict[int, "AgentPolicy"] = {}

    def reset(self, simulation: "Simulation" = None) -> None:
        """Reset all agent states.

        Args:
            simulation: The Simulation object (needed for grid_objects access)
        """
        if simulation is None:
            raise RuntimeError("CoordinatingPolicy requires simulation parameter in reset()")

        self._sim = simulation
        self._impl = CoordinatingPolicyImpl(simulation)
        self._agent_policies.clear()

    def agent_policy(self, agent_id: int):
        """Get an AgentPolicy instance for a specific agent."""
        if self._impl is None:
            raise RuntimeError("Policy not initialized - call reset(simulation=sim) first")

        # Create agent policies lazily
        if agent_id not in self._agent_policies:
            from cogames.policy import StatefulAgentPolicy

            self._agent_policies[agent_id] = StatefulAgentPolicy(self._impl, agent_id)
        return self._agent_policies[agent_id]
