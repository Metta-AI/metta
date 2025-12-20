"""
Harvest Policy - A focused scripted agent for the harvest mission.

This policy autonomously performs the harvest mission:
1. Gather resources from corner extractors
2. Craft hearts at the assembler
3. Deposit hearts in the chest
4. Recharge energy at the charger

Designed as a readable reference implementation for competing against neural network policies.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from mettagrid.policy.policy import MultiAgentPolicy, StatefulAgentPolicy, StatefulPolicyImpl
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action
from mettagrid.simulator.interface import AgentObservation

from .pathfinding import compute_goal_cells, shortest_path
from .pathfinding import is_traversable as path_is_traversable
from .pathfinding import is_within_bounds as path_is_within_bounds
from .types import CellType, ExtractorInfo, ObjectState, ParsedObservation
from .utils import (
    change_vibe_action,
    is_adjacent,
    is_station,
    is_wall,
    parse_observation,
    read_inventory_from_obs,
    update_agent_position,
    use_object_at,
)


class HarvestPhase(Enum):
    """Phases for the harvest mission."""

    GATHER = "gather"  # Collect resources from extractors
    ASSEMBLE = "assemble"  # Craft hearts at the assembler
    DELIVER = "deliver"  # Deposit hearts in the chest
    RECHARGE = "recharge"  # Restore energy at the charger


@dataclass
class HarvestState:
    """State for a harvest agent."""

    agent_id: int

    # Current phase
    phase: HarvestPhase = HarvestPhase.GATHER
    step_count: int = 0

    # Position (relative to starting position)
    row: int = 0
    col: int = 0
    energy: int = 100

    # Inventory
    carbon: int = 0
    oxygen: int = 0
    germanium: int = 0
    silicon: int = 0
    hearts: int = 0

    # Discovered objects
    extractors: dict[str, list[ExtractorInfo]] = field(
        default_factory=lambda: {"carbon": [], "oxygen": [], "germanium": [], "silicon": []}
    )
    stations: dict[str, tuple[int, int] | None] = field(
        default_factory=lambda: {"assembler": None, "chest": None, "charger": None}
    )

    # Heart crafting recipe (discovered from assembler)
    heart_recipe: Optional[dict[str, int]] = None

    # Map state
    map_height: int = 0
    map_width: int = 0
    occupancy: list[list[int]] = field(default_factory=list)
    agent_occupancy: set[tuple[int, int]] = field(default_factory=set)

    # Navigation
    last_action: Action = field(default_factory=lambda: Action(name="noop"))
    current_glyph: str = "default"
    using_object_this_step: bool = False

    # Path caching
    cached_path: Optional[list[tuple[int, int]]] = None
    cached_path_target: Optional[tuple[int, int]] = None
    cached_path_reach_adjacent: bool = False

    # Exploration state
    exploration_direction: Optional[str] = None
    exploration_step: int = 0
    explored_cells: set[tuple[int, int]] = field(default_factory=set)

    # Current observation
    current_obs: Optional[AgentObservation] = None


# Resource to vibe mapping for extracting resources
RESOURCE_VIBES = {
    "carbon": "carbon_a",
    "oxygen": "oxygen_a",
    "germanium": "germanium_a",
    "silicon": "silicon_a",
}


class HarvestAgentPolicy(StatefulPolicyImpl[HarvestState]):
    """Single-agent policy implementation for the harvest mission."""

    def __init__(self, policy_env_info: PolicyEnvInterface, agent_id: int):
        self._agent_id = agent_id
        self._policy_env_info = policy_env_info

        # Observation grid dimensions
        self._obs_hr = policy_env_info.obs_height // 2
        self._obs_wr = policy_env_info.obs_width // 2

        # Action lookup
        self._actions = policy_env_info.actions
        self._move_deltas = {
            "north": (-1, 0),
            "south": (1, 0),
            "east": (0, 1),
            "west": (0, -1),
        }

        # Feature parsing config
        self._spatial_feature_names = {"tag", "cooldown_remaining", "clipped", "remaining_uses"}
        self._agent_feature_key_by_name = {"agent:group": "agent_group", "agent:frozen": "agent_frozen"}
        self._protocol_input_prefix = "protocol_input:"
        self._protocol_output_prefix = "protocol_output:"

        # Energy thresholds
        self._recharge_low = 35
        self._recharge_high = 85

    def initial_agent_state(self) -> HarvestState:
        """Initialize state for the agent."""
        self._tag_names = self._policy_env_info.tag_id_to_name

        # Initialize heart recipe from assembler protocols
        heart_recipe = None
        for protocol in self._policy_env_info.assembler_protocols:
            if protocol.output_resources.get("heart", 0) > 0:
                heart_recipe = dict(protocol.input_resources)
                heart_recipe.pop("energy", None)  # Energy is not a gatherable resource
                break

        map_size = 200
        center = map_size // 2

        return HarvestState(
            agent_id=self._agent_id,
            map_height=map_size,
            map_width=map_size,
            occupancy=[[CellType.FREE.value] * map_size for _ in range(map_size)],
            row=center,
            col=center,
            heart_recipe=heart_recipe,
        )

    def step_with_state(self, obs: AgentObservation, state: HarvestState) -> tuple[Action, HarvestState]:
        """Main step function - process observation and return action."""
        state.step_count += 1
        state.current_obs = obs
        state.agent_occupancy.clear()

        # Update inventory from observation
        read_inventory_from_obs(state, obs, obs_hr=self._obs_hr, obs_wr=self._obs_wr)

        # Update position based on last action
        update_agent_position(state, move_deltas=self._move_deltas)

        # Parse observation to discover objects
        parsed = self._parse_observation(state, obs)
        self._discover_objects(state, parsed)

        # Determine phase
        self._update_phase(state)

        # Set vibe for current phase
        desired_vibe = self._get_vibe_for_phase(state)
        if state.current_glyph != desired_vibe:
            state.current_glyph = desired_vibe
            action = change_vibe_action(desired_vibe, actions=self._actions)
            state.last_action = action
            return action, state

        # Execute current phase
        action = self._execute_phase(state)
        state.last_action = action
        return action, state

    def _parse_observation(self, state: HarvestState, obs: AgentObservation) -> ParsedObservation:
        """Parse observation into structured format."""
        return parse_observation(
            state,
            obs,
            obs_hr=self._obs_hr,
            obs_wr=self._obs_wr,
            spatial_feature_names=self._spatial_feature_names,
            agent_feature_key_by_name=self._agent_feature_key_by_name,
            protocol_input_prefix=self._protocol_input_prefix,
            protocol_output_prefix=self._protocol_output_prefix,
            tag_names=self._tag_names,
        )

    def _discover_objects(self, state: HarvestState, parsed: ParsedObservation) -> None:
        """Update occupancy map and discover extractors/stations."""
        if state.row < 0:
            return

        # Mark all observed cells as FREE initially and track as explored
        for obs_r in range(2 * self._obs_hr + 1):
            for obs_c in range(2 * self._obs_wr + 1):
                r = obs_r - self._obs_hr + state.row
                c = obs_c - self._obs_wr + state.col
                if 0 <= r < state.map_height and 0 <= c < state.map_width:
                    state.occupancy[r][c] = CellType.FREE.value
                    state.explored_cells.add((r, c))

        # Process visible objects
        for pos, obj_state in parsed.nearby_objects.items():
            r, c = pos
            obj_name = obj_state.name.lower()

            # Walls are obstacles
            if is_wall(obj_name):
                state.occupancy[r][c] = CellType.OBSTACLE.value
                continue

            # Track other agents
            if obj_name == "agent":
                state.agent_occupancy.add((r, c))
                continue

            # Discover stations
            for station_name in ("assembler", "chest", "charger"):
                if is_station(obj_name, station_name):
                    state.occupancy[r][c] = CellType.OBSTACLE.value
                    if state.stations.get(station_name) is None:
                        state.stations[station_name] = pos
                    break
            else:
                # Discover extractors
                if "extractor" in obj_name:
                    state.occupancy[r][c] = CellType.OBSTACLE.value
                    resource_type = obj_name.replace("_extractor", "").replace("clipped_", "")
                    if resource_type in state.extractors:
                        self._update_extractor(state, pos, resource_type, obj_state)

    def _update_extractor(
        self, state: HarvestState, pos: tuple[int, int], resource_type: str, obj_state: ObjectState
    ) -> None:
        """Update or create extractor info."""
        # Find existing extractor at this position
        extractor = None
        for ext in state.extractors[resource_type]:
            if ext.position == pos:
                extractor = ext
                break

        if extractor is None:
            extractor = ExtractorInfo(
                position=pos,
                resource_type=resource_type,
                last_seen_step=state.step_count,
            )
            state.extractors[resource_type].append(extractor)

        extractor.last_seen_step = state.step_count
        extractor.cooldown_remaining = obj_state.cooldown_remaining
        extractor.clipped = obj_state.clipped > 0
        extractor.remaining_uses = obj_state.remaining_uses

    def _update_phase(self, state: HarvestState) -> None:
        """Update phase based on current state."""
        # Priority 1: Recharge if energy low
        if state.energy < self._recharge_low:
            state.phase = HarvestPhase.RECHARGE
            return

        # Stay in recharge until energy restored
        if state.phase == HarvestPhase.RECHARGE and state.energy < self._recharge_high:
            return

        # Priority 2: Deliver hearts
        if state.hearts > 0:
            state.phase = HarvestPhase.DELIVER
            return

        # Priority 3: Assemble if we have all resources
        if state.heart_recipe and self._can_assemble(state):
            state.phase = HarvestPhase.ASSEMBLE
            return

        # Priority 4: Gather resources
        state.phase = HarvestPhase.GATHER

    def _can_assemble(self, state: HarvestState) -> bool:
        """Check if we have resources for a heart."""
        if state.heart_recipe is None:
            return False
        return (
            state.carbon >= state.heart_recipe.get("carbon", 0)
            and state.oxygen >= state.heart_recipe.get("oxygen", 0)
            and state.germanium >= state.heart_recipe.get("germanium", 0)
            and state.silicon >= state.heart_recipe.get("silicon", 0)
        )

    def _get_vibe_for_phase(self, state: HarvestState) -> str:
        """Get the appropriate vibe for the current phase."""
        if state.phase == HarvestPhase.ASSEMBLE:
            return "heart_a"
        elif state.phase == HarvestPhase.DELIVER:
            return "heart_b"  # Deposit hearts to chest (heart_b transfers +1 heart)
        elif state.phase == HarvestPhase.RECHARGE:
            return "charger"
        else:
            # GATHER: use heart_a vibe to coordinate with other agents for vibe_check missions
            # (vibe_check requires 2+ agents on heart_a to craft hearts)
            return "heart_a"

    def _execute_phase(self, state: HarvestState) -> Action:
        """Execute action for current phase."""
        if state.phase == HarvestPhase.GATHER:
            return self._do_gather(state)
        elif state.phase == HarvestPhase.ASSEMBLE:
            return self._do_assemble(state)
        elif state.phase == HarvestPhase.DELIVER:
            return self._do_deliver(state)
        elif state.phase == HarvestPhase.RECHARGE:
            return self._do_recharge(state)
        return self._actions.noop.Noop()

    def _do_gather(self, state: HarvestState) -> Action:
        """Gather resources from extractors."""
        # Find which resources we still need
        deficits = self._calculate_deficits(state)
        if all(d <= 0 for d in deficits.values()):
            return self._actions.noop.Noop()

        # Find an extractor for a needed resource
        extractor = self._find_needed_extractor(state, deficits)

        if extractor is None:
            # Need to explore to find extractors
            return self._explore(state)

        # Navigate to extractor
        adj = is_adjacent((state.row, state.col), extractor.position)
        if not adj:
            return self._move_towards(state, extractor.position, reach_adjacent=True)

        # Adjacent - use it
        if extractor.cooldown_remaining > 0 or extractor.remaining_uses == 0 or extractor.clipped:
            return self._actions.noop.Noop()

        return use_object_at(
            state, extractor.position, actions=self._actions, move_deltas=self._move_deltas, using_for="extractor"
        )

    def _do_assemble(self, state: HarvestState) -> Action:
        """Assemble hearts at the assembler."""
        assembler = state.stations.get("assembler")

        if assembler is None:
            return self._explore(state)

        if not is_adjacent((state.row, state.col), assembler):
            return self._move_towards(state, assembler, reach_adjacent=True)

        return use_object_at(
            state, assembler, actions=self._actions, move_deltas=self._move_deltas, using_for="assembler"
        )

    def _do_deliver(self, state: HarvestState) -> Action:
        """Deliver hearts to the chest."""
        chest = state.stations.get("chest")

        if chest is None:
            return self._explore(state)

        if not is_adjacent((state.row, state.col), chest):
            return self._move_towards(state, chest, reach_adjacent=True)

        return use_object_at(state, chest, actions=self._actions, move_deltas=self._move_deltas, using_for="chest")

    def _do_recharge(self, state: HarvestState) -> Action:
        """Recharge at the charger."""
        charger = state.stations.get("charger")

        if charger is None:
            return self._explore(state)

        if not is_adjacent((state.row, state.col), charger):
            return self._move_towards(state, charger, reach_adjacent=True)

        return use_object_at(state, charger, actions=self._actions, move_deltas=self._move_deltas, using_for="charger")

    def _calculate_deficits(self, state: HarvestState) -> dict[str, int]:
        """Calculate how many more resources we need."""
        if state.heart_recipe is None:
            return {"carbon": 1, "oxygen": 1, "germanium": 1, "silicon": 1}

        return {
            "carbon": max(0, state.heart_recipe.get("carbon", 0) - state.carbon),
            "oxygen": max(0, state.heart_recipe.get("oxygen", 0) - state.oxygen),
            "germanium": max(0, state.heart_recipe.get("germanium", 0) - state.germanium),
            "silicon": max(0, state.heart_recipe.get("silicon", 0) - state.silicon),
        }

    def _find_needed_extractor(self, state: HarvestState, deficits: dict[str, int]) -> Optional[ExtractorInfo]:
        """Find the nearest available extractor for a needed resource."""
        # Sort by deficit size (largest first)
        for resource, deficit in sorted(deficits.items(), key=lambda x: x[1], reverse=True):
            if deficit > 0:
                extractor = self._find_nearest_extractor(state, resource)
                if extractor is not None:
                    return extractor
        return None

    def _find_nearest_extractor(self, state: HarvestState, resource_type: str) -> Optional[ExtractorInfo]:
        """Find the nearest available extractor of a given type."""
        extractors = state.extractors.get(resource_type, [])
        available = [e for e in extractors if not e.clipped and e.remaining_uses > 0]

        if not available:
            return None

        def distance(ext: ExtractorInfo) -> int:
            return abs(ext.position[0] - state.row) + abs(ext.position[1] - state.col)

        return min(available, key=distance)

    def _move_towards(self, state: HarvestState, target: tuple[int, int], reach_adjacent: bool = False) -> Action:
        """Pathfind towards a target using BFS."""
        start = (state.row, state.col)

        if start == target and not reach_adjacent:
            return self._actions.noop.Noop()

        goal_cells = compute_goal_cells(state, target, reach_adjacent, CellType)
        if not goal_cells:
            return self._actions.noop.Noop()

        # Check cached path
        if (
            state.cached_path
            and state.cached_path_target == target
            and state.cached_path_reach_adjacent == reach_adjacent
        ):
            if state.cached_path:
                next_pos = state.cached_path[0]
                if path_is_traversable(state, next_pos[0], next_pos[1], CellType):
                    path = state.cached_path
                else:
                    path = None
            else:
                path = None
        else:
            path = None

        # Compute new path if needed
        if path is None:
            path = shortest_path(state, start, goal_cells, False, CellType)
            state.cached_path = path.copy() if path else None
            state.cached_path_target = target
            state.cached_path_reach_adjacent = reach_adjacent

        if not path:
            return self._explore(state)

        # Get next step
        next_pos = path[0]
        state.cached_path = path[1:] if len(path) > 1 else None

        # Convert to action
        dr = next_pos[0] - state.row
        dc = next_pos[1] - state.col

        if dr == -1 and dc == 0:
            return self._actions.move.Move("north")
        elif dr == 1 and dc == 0:
            return self._actions.move.Move("south")
        elif dr == 0 and dc == 1:
            return self._actions.move.Move("east")
        elif dr == 0 and dc == -1:
            return self._actions.move.Move("west")

        return self._actions.noop.Noop()

    def _explore(self, state: HarvestState) -> Action:
        """Explore using frontier-based navigation for better maze coverage."""
        # Find frontier cells: explored cells adjacent to unexplored passable cells
        frontier_targets = self._find_frontier_targets(state)

        if frontier_targets:
            # Navigate to nearest frontier cell using direct BFS (avoid recursion via _move_towards)
            nearest = min(
                frontier_targets,
                key=lambda pos: abs(pos[0] - state.row) + abs(pos[1] - state.col),
            )
            start = (state.row, state.col)
            goal_cells = {nearest}
            path = shortest_path(state, start, goal_cells, False, CellType)

            if path:
                next_pos = path[0]
                dr = next_pos[0] - state.row
                dc = next_pos[1] - state.col
                if dr == -1:
                    return self._actions.move.Move("north")
                elif dr == 1:
                    return self._actions.move.Move("south")
                elif dc == 1:
                    return self._actions.move.Move("east")
                elif dc == -1:
                    return self._actions.move.Move("west")

        # Fallback: random direction exploration for when stuck or no frontier
        if state.exploration_direction is None or state.step_count - state.exploration_step > 15:
            directions = ["north", "south", "east", "west"]
            random.shuffle(directions)

            for direction in directions:
                dr, dc = self._move_deltas[direction]
                nr, nc = state.row + dr, state.col + dc
                if path_is_within_bounds(state, nr, nc) and path_is_traversable(state, nr, nc, CellType):
                    state.exploration_direction = direction
                    state.exploration_step = state.step_count
                    break

        if state.exploration_direction:
            dr, dc = self._move_deltas[state.exploration_direction]
            nr, nc = state.row + dr, state.col + dc
            if path_is_within_bounds(state, nr, nc) and path_is_traversable(state, nr, nc, CellType):
                return self._actions.move.Move(state.exploration_direction)

        # Blocked - pick a new direction
        state.exploration_direction = None
        return self._actions.noop.Noop()

    def _find_frontier_targets(self, state: HarvestState) -> list[tuple[int, int]]:
        """Find explored cells that are adjacent to unexplored cells (frontier)."""
        frontiers = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for r, c in state.explored_cells:
            # Skip if not traversable (can't stand there)
            if not path_is_traversable(state, r, c, CellType):
                continue

            # Check if any neighbor is unexplored
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if path_is_within_bounds(state, nr, nc) and (nr, nc) not in state.explored_cells:
                    # This is a frontier cell - we can stand here and see unexplored areas
                    frontiers.append((r, c))
                    break

        return frontiers


class HarvestPolicy(MultiAgentPolicy):
    """Multi-agent policy wrapper for the harvest mission."""

    short_names = ["harvest", "harvest_policy"]

    def __init__(self, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info)
        self._agent_policies: dict[int, StatefulAgentPolicy[HarvestState]] = {}

    def agent_policy(self, agent_id: int) -> StatefulAgentPolicy[HarvestState]:
        if agent_id not in self._agent_policies:
            self._agent_policies[agent_id] = StatefulAgentPolicy(
                HarvestAgentPolicy(self._policy_env_info, agent_id),
                self._policy_env_info,
                agent_id=agent_id,
            )
        return self._agent_policies[agent_id]
