"""Policy wrapper for CoordinatingAgentImpl."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from cogames.policy.scripted_agent.simple_baseline_agent import CellType, ExtractorInfo, Phase
from cogames.policy.scripted_agent.unclipping_policy import UnclippingAgentImpl, UnclippingAgentState
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy, StatefulAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action, AgentObservation, Simulation


@dataclass
class CoordinatingAgentState(UnclippingAgentState):
    """Extended state for coordinating agent."""

    # Track home base for yielding
    home_base_row: int = -1
    home_base_col: int = -1

    # Track reserved adjacent cell for extractor
    reserved_adjacent_cell: tuple[int, int] | None = None


class CoordinatingAgentImpl(UnclippingAgentImpl):
    """Internal implementation for coordinating agent with multi-agent coordination."""

    # Shared state across all agent instances (class-level)
    _assembly_signal: dict = {"active": False, "requester": None, "position": None}
    _assembly_signal_participants: set[int] = set()

    # Target reservation: track which extractor AND which adjacent cell
    _target_assignments: dict[int, tuple[int, int]] = {}  # agent_id -> extractor_position
    _target_position_counts: dict[tuple[int, int], int] = {}  # extractor_position -> count
    _reserved_adjacent_cells: dict[
        tuple[int, int], set[tuple[int, int]]
    ] = {}  # extractor_pos -> set of reserved adjacent cells

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        agent_id: int,
        simulation: Optional[Simulation],
    ):
        """Initialize coordinating agent implementation."""
        super().__init__(policy_env_info, agent_id, simulation)
        print(
            "[CoordinatingAgent] Initialized with assembly coordination "
            "and extractor sharing (up to 4 agents/extractor)"
        )

    def agent_state(self, simulation: Simulation | None = None) -> CoordinatingAgentState:
        """Get initial state for coordinating agent."""
        if simulation is None:
            return CoordinatingAgentState(
                agent_id=self._agent_id,
                id_map=None,
                map_height=0,
                map_width=0,
                occupancy=None,
                simulation=None,
            )

        map_height = simulation.map_height
        map_width = simulation.map_width
        occupancy = [[CellType.FREE.value] * map_width for _ in range(map_height)]
        print(f"[CoordinatingAgent] Initialized for map {map_height}x{map_width}")
        return CoordinatingAgentState(
            agent_id=self._agent_id,
            id_map=simulation.id_map,
            simulation=simulation,
            map_height=map_height,
            map_width=map_width,
            occupancy=occupancy,
        )

    def step_with_state(
        self, obs: AgentObservation, state: CoordinatingAgentState
    ) -> tuple[Action, CoordinatingAgentState]:
        """Get action and update state."""
        action = self._step_impl(state, obs)
        return action, state

    def _get_adjacent_cells(self, pos: tuple[int, int]) -> list[tuple[int, int]]:
        """Get all 4 adjacent cells for a position."""
        r, c = pos
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]

    def _find_unreserved_adjacent_cell(
        self, s: CoordinatingAgentState, extractor_pos: tuple[int, int]
    ) -> tuple[int, int] | None:
        """Find an unreserved adjacent cell around an extractor."""
        reserved = self._reserved_adjacent_cells.get(extractor_pos, set())
        adjacent_cells = self._get_adjacent_cells(extractor_pos)

        # Filter out reserved cells and cells that are obstacles
        available_cells = []
        for cell in adjacent_cells:
            if cell in reserved:
                continue
            r, c = cell
            # Check if valid and passable
            if 0 <= r < s.map_height and 0 <= c < s.map_width:
                if self._is_passable(s, r, c):
                    available_cells.append(cell)

        if not available_cells:
            return None

        # Pick the cell closest to agent's current position
        def distance(cell: tuple[int, int]) -> int:
            return abs(cell[0] - s.row) + abs(cell[1] - s.col)

        return min(available_cells, key=distance)

    def _set_agent_target(
        self, agent_id: int, target: tuple[int, int], s: CoordinatingAgentState
    ) -> tuple[int, int] | None:
        """Reserve a target extractor and specific adjacent cell for an agent.

        Returns the reserved adjacent cell, or None if no cell available.
        """
        # Clear previous target
        prev = self._target_assignments.get(agent_id)
        if prev is not None and prev != target:
            count = self._target_position_counts.get(prev, 0)
            if count > 1:
                self._target_position_counts[prev] = count - 1
            else:
                self._target_position_counts.pop(prev, None)

            # Clear reserved adjacent cells for previous extractor
            # (We don't track which specific cell was this agent's, so we'll just
            # re-reserve when they target this extractor again)
            if prev in self._reserved_adjacent_cells and not self._target_position_counts.get(prev, 0):
                # No more agents targeting this extractor, clear all reservations
                self._reserved_adjacent_cells.pop(prev, None)

        # Find an unreserved adjacent cell
        adjacent_cell = self._find_unreserved_adjacent_cell(s, target)
        if adjacent_cell is None:
            return None  # No available cells

        # Reserve the extractor
        self._target_assignments[agent_id] = target
        self._target_position_counts[target] = self._target_position_counts.get(target, 0) + 1

        # Reserve the adjacent cell
        if target not in self._reserved_adjacent_cells:
            self._reserved_adjacent_cells[target] = set()
        self._reserved_adjacent_cells[target].add(adjacent_cell)

        return adjacent_cell

    def _clear_agent_target(self, agent_id: int) -> None:
        """Release a target reservation."""
        prev = self._target_assignments.pop(agent_id, None)
        if prev is not None:
            count = self._target_position_counts.get(prev, 0)
            if count > 1:
                self._target_position_counts[prev] = count - 1
            else:
                self._target_position_counts.pop(prev, None)

    def _target_count(self, position: tuple[int, int]) -> int:
        """Get number of agents currently targeting this position."""
        return self._target_position_counts.get(position, 0)

    def _clear_assembly_signal(self) -> None:
        """Clear the assembly coordination signal."""
        self._assembly_signal = {"active": False, "requester": None, "position": None}
        self._assembly_signal_participants.clear()

    def _update_phase(self, s: CoordinatingAgentState) -> None:
        """Override to add assembly coordination logic."""
        # Priority 1: Recharge if energy low
        if s.energy < 30:
            if s.phase != Phase.RECHARGE:
                print(f"[Agent {s.agent_id}] Phase: {s.phase.name} -> RECHARGE (energy={s.energy})")
                s.phase = Phase.RECHARGE
            return

        # Stay in RECHARGE until energy is fully restored (>= 90)
        if s.phase == Phase.RECHARGE:
            if s.energy >= 90:
                print(f"[Agent {s.agent_id}] Phase: RECHARGE -> GATHER (energy={s.energy})")
                s.phase = Phase.GATHER
                s.target_position = None
            return

        # Priority 2: Deliver hearts if we have any
        if s.hearts > 0:
            if s.phase != Phase.DELIVER:
                print(f"[Agent {s.agent_id}] Phase: {s.phase.name} -> DELIVER ({s.hearts} hearts)")
                s.phase = Phase.DELIVER
            return

        # Priority 3: Assemble if we have all resources (with coordination)
        can_assemble = (
            s.carbon >= self._heart_recipe["carbon"]
            and s.oxygen >= self._heart_recipe["oxygen"]
            and s.germanium >= self._heart_recipe["germanium"]
            and s.silicon >= self._heart_recipe["silicon"]
        )

        if can_assemble:
            # Assembly coordination: Only one agent at assembler at a time
            # (Assembler can only be used by one agent, unlike extractors)
            if self._assembly_signal["active"]:
                requester = self._assembly_signal["requester"]
                if requester != s.agent_id:
                    # Another agent is assembling - continue gathering or wait
                    if s.phase != Phase.GATHER:
                        print(
                            f"[Agent {s.agent_id}] Phase: {s.phase.name} -> GATHER "
                            f"(yielding assembly to agent {requester})"
                        )
                        s.phase = Phase.GATHER
                        # Clear target so agent finds something else to do
                        self._clear_agent_target(s.agent_id)
                    return
                # This agent is the requester, proceed to assemble
            else:
                # No one assembling, claim the signal
                self._assembly_signal = {
                    "active": True,
                    "requester": s.agent_id,
                    "position": s.stations.get("assembler"),
                }
                print(f"[Agent {s.agent_id}] Claimed assembly (other agents can use extractors)")

            if s.phase != Phase.ASSEMBLE:
                print(f"[Agent {s.agent_id}] Phase: {s.phase.name} -> ASSEMBLE (all resources ready)")
                s.phase = Phase.ASSEMBLE
            return

        # If we were assembling but don't have resources anymore, clear signal
        if s.phase == Phase.ASSEMBLE and not can_assemble:
            if self._assembly_signal.get("requester") == s.agent_id:
                self._clear_assembly_signal()
                print(f"[Agent {s.agent_id}] Released assembly signal (missing resources)")

        # Priority 4: Unclip if blocked and have unclip item
        if s.blocked_by_clipped_extractor is not None and self._has_unclip_item(s):
            if s.phase != Phase.UNCLIP:
                print(
                    f"[Agent {s.agent_id}] Phase: {s.phase.name} -> UNCLIP "
                    f"(have unclip item for {s.unclip_target_resource})"
                )
                s.phase = Phase.UNCLIP
            return

        # Priority 5: Craft unclip item if blocked but don't have item
        if s.blocked_by_clipped_extractor is not None and not self._has_unclip_item(s):
            if s.phase != Phase.CRAFT_UNCLIP:
                print(
                    f"[Agent {s.agent_id}] Phase: {s.phase.name} -> CRAFT_UNCLIP "
                    f"(need to craft for {s.unclip_target_resource})"
                )
                s.phase = Phase.CRAFT_UNCLIP
            return

        # Priority 6: Default to GATHER
        if s.phase != Phase.GATHER:
            print(f"[Agent {s.agent_id}] Phase: {s.phase.name} -> GATHER (need resources)")
            s.phase = Phase.GATHER
            s.target_position = None

    def _find_any_needed_extractor(self, s: CoordinatingAgentState) -> Optional[tuple[ExtractorInfo, str]]:
        """
        Override to prefer extractors with fewer agents targeting them.

        This helps distribute agents across multiple extractors.
        """
        deficits = self._calculate_deficits(s)

        for resource_type in ["carbon", "oxygen", "germanium", "silicon"]:
            if deficits.get(resource_type, 0) <= 0:
                continue

            extractors = s.extractors.get(resource_type, [])
            if not extractors:
                continue

            # Filter available
            if s.unreachable_extractors is None:
                s.unreachable_extractors = {}

            available = [
                e
                for e in extractors
                if not e.clipped and e.remaining_uses > 0 and s.unreachable_extractors.get(e.position, 0) < 5
            ]

            if available:
                # Sort by distance only - agents can share extractors
                # Up to 4 agents can use an extractor from different adjacent positions
                def distance(pos: tuple[int, int]) -> int:
                    return abs(pos[0] - s.row) + abs(pos[1] - s.col)

                # Only avoid extractors that are fully saturated (4+ agents)
                def sort_key(e):
                    target_count = self._target_count(e.position)
                    dist = distance(e.position)
                    # Heavy penalty only if extractor is fully saturated
                    saturation_penalty = 1000 if target_count >= 4 else 0
                    return (saturation_penalty, dist)

                nearest = min(available, key=sort_key)

                # Reserve this target and get reserved adjacent cell
                adjacent_cell = self._set_agent_target(s.agent_id, nearest.position, s)
                if adjacent_cell is None:
                    # No adjacent cells available, extractor is fully surrounded
                    continue

                # Store reserved cell in state
                s.reserved_adjacent_cell = adjacent_cell
                print(
                    f"[Agent {s.agent_id}] Reserved {nearest.resource_type} extractor at {nearest.position}, "
                    f"adjacent cell {adjacent_cell}"
                )

                return (nearest, resource_type)

            # No available extractors - check if any are clipped or depleted
            clipped = [e for e in extractors if e.clipped or e.remaining_uses == 0]
            if clipped:

                def distance(pos: tuple[int, int]) -> int:
                    return abs(pos[0] - s.row) + abs(pos[1] - s.col)

                nearest_clipped = min(clipped, key=lambda e: distance(e.position))
                s.blocked_by_clipped_extractor = nearest_clipped.position
                s.unclip_target_resource = resource_type
                print(
                    f"[Agent {s.agent_id}] All {resource_type} extractors clipped! "
                    f"Need to unclip at {nearest_clipped.position}"
                )
                return None

        return None

    def _move_towards(
        self,
        s: CoordinatingAgentState,
        target: tuple[int, int],
        *,
        reach_adjacent: bool = False,
        allow_goal_block: bool = False,
    ) -> Action:
        """Override to use reserved adjacent cell when moving to extractors."""
        # If we have a reserved adjacent cell and we're moving to an extractor with reach_adjacent=True,
        # instead move directly to the reserved cell
        if reach_adjacent and s.reserved_adjacent_cell is not None:
            # Check if this target is an extractor we have a reservation for
            if self._target_assignments.get(s.agent_id) == target:
                # Move directly to the reserved cell instead
                target = s.reserved_adjacent_cell
                reach_adjacent = False  # We want to reach this exact cell
                print(
                    f"[Agent {s.agent_id}] Using reserved adjacent cell {target} "
                    f"for extractor at {self._target_assignments.get(s.agent_id)}"
                )

        # Call parent implementation with keyword arguments
        return super()._move_towards(s, target, reach_adjacent=reach_adjacent, allow_goal_block=allow_goal_block)

    def _update_state_from_obs(self, s: CoordinatingAgentState, obs: AgentObservation) -> None:
        """Override to track home base and clear assembly signal when heart received."""
        # Call parent to update state
        super()._update_state_from_obs(s, obs)

        # Track home base (first position)
        if s.home_base_row == -1 and s.row >= 0:
            s.home_base_row = s.row
            s.home_base_col = s.col

        # Clear assembly signal when heart received
        if s.hearts > 0 and self._assembly_signal.get("requester") == s.agent_id:
            self._clear_assembly_signal()
            print(f"[Agent {s.agent_id}] Released assembly signal (heart received)")


class CoordinatingPolicy(MultiAgentPolicy):
    """Policy class for coordinating agent."""

    def __init__(self, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info)
        self._policy_env_info = policy_env_info
        self._agent_policies: list[AgentPolicy] = [
            StatefulAgentPolicy(CoordinatingAgentImpl(self._policy_env_info, agent_id, simulation=None), agent_id)
            for agent_id in range(self._policy_env_info.num_agents)
        ]

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        """Get policy for a specific agent."""
        return self._agent_policies[agent_id]
