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
from .pathfinding_fast import PathCache, compute_goal_cells_fast
from .types import CellType, ExtractorInfo, ObjectState, ParsedObservation
from .utils import (
    change_vibe_action,
    is_adjacent,
    is_station,
    is_wall,
    parse_observation,
    read_inventory_from_obs,
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

    # Pathfinding cache (initialized after map_size is known)
    path_cache: Optional[PathCache] = None

    # Move verification - store previous observation landmarks to detect move failure
    prev_landmark_obs_pos: Optional[tuple[int, int]] = None  # obs-relative position of landmark
    prev_landmark_tag: Optional[int] = None  # tag ID of the landmark object


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
            path_cache=PathCache(map_size, map_size),
        )

    def step_with_state(self, obs: AgentObservation, state: HarvestState) -> tuple[Action, HarvestState]:
        """Main step function - process observation and return action."""
        state.step_count += 1
        state.current_obs = obs
        state.agent_occupancy.clear()

        # Update inventory from observation
        read_inventory_from_obs(state, obs, obs_hr=self._obs_hr, obs_wr=self._obs_wr)

        # Update position based on last action - but verify move success first!
        self._update_position_with_verification(state, obs)

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

    def _update_position_with_verification(self, state: HarvestState, obs: AgentObservation) -> None:
        """Update position only if move actually succeeded.

        Uses landmark comparison to detect move failure:
        - Before a move, we store a landmark (a visible wall or obstacle)
        - After the move, we check if the landmark shifted by the expected amount
        - If not, the move failed and we don't update position
        """
        # If last action wasn't a move, just clear flags
        if not state.last_action or not state.last_action.name.startswith("move_"):
            state.using_object_this_step = False
            self._store_landmark(state, obs)
            return

        # If we were using an object, position doesn't change
        if state.using_object_this_step:
            state.using_object_this_step = False
            self._store_landmark(state, obs)
            return

        # Get expected movement delta
        direction = state.last_action.name[5:]  # Remove "move_" prefix
        expected_delta = self._move_deltas.get(direction)
        if expected_delta is None:
            self._store_landmark(state, obs)
            return

        dr, dc = expected_delta

        # Check if move succeeded by verifying landmark position shift
        move_succeeded = self._verify_move_success(state, obs, dr, dc)

        if move_succeeded:
            state.row += dr
            state.col += dc
            # Invalidate cached path since we moved
            state.cached_path = None

        # Store new landmark for next step
        self._store_landmark(state, obs)

    def _verify_move_success(self, state: HarvestState, obs: AgentObservation, dr: int, dc: int) -> bool:
        """Verify if the last move succeeded by checking landmark position.

        If we moved north (dr=-1), objects should shift south in obs coords (obs_row += 1).
        """
        if state.prev_landmark_obs_pos is None or state.prev_landmark_tag is None:
            # No previous landmark - assume move succeeded (first step)
            return True

        prev_r, prev_c = state.prev_landmark_obs_pos
        prev_tag = state.prev_landmark_tag

        # Expected new position of landmark in observation coords
        # Movement is opposite in observation space
        expected_new_r = prev_r + dr
        expected_new_c = prev_c + dc

        # Find the landmark in current observation
        for tok in obs.tokens:
            if tok.feature.name == "tag" and tok.value == prev_tag:
                obs_r, obs_c = tok.location
                # If landmark is at expected position, move succeeded
                if obs_r == expected_new_r and obs_c == expected_new_c:
                    return True
                # If landmark is at same position as before, move failed
                if obs_r == prev_r and obs_c == prev_c:
                    return False

        # Landmark not found (might have moved out of view) - assume move succeeded
        return True

    def _store_landmark(self, state: HarvestState, obs: AgentObservation) -> None:
        """Store a stable landmark from current observation for move verification.

        We prefer walls since they never move. Look for walls near the edge of vision
        so they're likely to still be visible after a move.
        """
        best_landmark = None
        best_distance = 0

        center_r, center_c = self._obs_hr, self._obs_wr

        for tok in obs.tokens:
            if tok.feature.name != "tag":
                continue

            obs_r, obs_c = tok.location
            # Skip center (that's us)
            if obs_r == center_r and obs_c == center_c:
                continue

            tag_name = self._tag_names.get(tok.value, "")
            # Prefer walls as landmarks (they never move)
            if "wall" in tag_name.lower():
                # Prefer landmarks near edge of vision but not at the very edge
                dist_from_center = abs(obs_r - center_r) + abs(obs_c - center_c)
                if dist_from_center > best_distance and dist_from_center < self._obs_hr:
                    best_distance = dist_from_center
                    best_landmark = (tok.location, tok.value)

        if best_landmark:
            state.prev_landmark_obs_pos = best_landmark[0]
            state.prev_landmark_tag = best_landmark[1]
        else:
            state.prev_landmark_obs_pos = None
            state.prev_landmark_tag = None

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
                        # Pre-compute distance map for this station
                        if state.path_cache is not None:
                            state.path_cache.compute_distance_map(
                                station_name,
                                pos,
                                state.occupancy,
                                CellType,
                                state.agent_occupancy,
                            )
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
            # GATHER: use vibe for target resource if known
            return "default"

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
        if not is_adjacent((state.row, state.col), extractor.position):
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
            return self._move_towards(state, assembler, reach_adjacent=True, station_name="assembler")

        return use_object_at(
            state, assembler, actions=self._actions, move_deltas=self._move_deltas, using_for="assembler"
        )

    def _do_deliver(self, state: HarvestState) -> Action:
        """Deliver hearts to the chest."""
        chest = state.stations.get("chest")

        if chest is None:
            return self._explore(state)

        if not is_adjacent((state.row, state.col), chest):
            return self._move_towards(state, chest, reach_adjacent=True, station_name="chest")

        return use_object_at(state, chest, actions=self._actions, move_deltas=self._move_deltas, using_for="chest")

    def _do_recharge(self, state: HarvestState) -> Action:
        """Recharge at the charger."""
        charger = state.stations.get("charger")

        if charger is None:
            return self._explore(state)

        if not is_adjacent((state.row, state.col), charger):
            return self._move_towards(state, charger, reach_adjacent=True, station_name="charger")

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

    def _move_towards(
        self, state: HarvestState, target: tuple[int, int], reach_adjacent: bool = False, station_name: str | None = None
    ) -> Action:
        """Pathfind towards a target using optimized BFS with caching.

        Args:
            target: Target position to reach
            reach_adjacent: If True, reach a cell adjacent to target instead of target itself
            station_name: If provided, try to use cached distance map for faster pathing
        """
        start = (state.row, state.col)

        if start == target and not reach_adjacent:
            return self._actions.noop.Noop()

        path: list[tuple[int, int]] | None = None

        # Try cached distance map for stations (O(path_length) instead of O(map_size) BFS)
        if station_name is not None and state.path_cache is not None and reach_adjacent:
            path = state.path_cache.get_path_to_station(
                station_name, start, state.occupancy, CellType, state.agent_occupancy
            )
            if path:
                # Distance map path found - use it
                state.cached_path = path[1:] if len(path) > 1 else None
                state.cached_path_target = target
                state.cached_path_reach_adjacent = reach_adjacent

        # Fall back to regular pathfinding
        if not path:
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

            # Compute new path if needed - use fast BFS if path_cache available
            if path is None:
                goal_cells = compute_goal_cells_fast(
                    state.occupancy, state.map_height, state.map_width,
                    state.agent_occupancy, target, reach_adjacent, CellType
                )
                if not goal_cells:
                    return self._actions.noop.Noop()

                if state.path_cache is not None:
                    path = state.path_cache.shortest_path_fast(
                        start, goal_cells, state.occupancy, CellType, state.agent_occupancy, False
                    )
                else:
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
        """Explore using frontier-based navigation for better maze coverage.

        Uses incremental frontier caching for O(new_cells) instead of O(all_explored_cells).
        """
        # Use cached frontiers if available (incremental update)
        if state.path_cache is not None:
            frontier_targets = list(state.path_cache.update_frontiers(
                state.explored_cells, state.occupancy, CellType
            ))
        else:
            frontier_targets = self._find_frontier_targets(state)

        if frontier_targets:
            # Navigate to nearest frontier cell using fast BFS
            nearest = min(
                frontier_targets,
                key=lambda pos: abs(pos[0] - state.row) + abs(pos[1] - state.col),
            )
            start = (state.row, state.col)
            goal_cells = [nearest]

            # Use fast BFS if available
            if state.path_cache is not None:
                path = state.path_cache.shortest_path_fast(
                    start, goal_cells, state.occupancy, CellType, state.agent_occupancy, False
                )
            else:
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

    short_names = []

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
