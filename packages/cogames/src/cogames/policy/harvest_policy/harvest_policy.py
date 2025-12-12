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

    # Move verification - store multiple landmarks for robust verification
    prev_landmarks: list[tuple[tuple[int, int], int]] = field(default_factory=list)  # [(obs_pos, tag_id), ...]

    # Drift detection - track consecutive failed moves
    consecutive_failed_moves: int = 0
    last_successful_move_step: int = 0

    # Stuck recovery - cycle through directions when stuck
    stuck_direction_idx: int = 0
    stuck_recovery_active: bool = False

    # Observation-based stuck detection (more reliable than landmark verification)
    prev_obs_hash: int = 0
    same_observation_count: int = 0

    # Exploration progress tracking
    last_exploration_progress_step: int = 0
    last_explored_count: int = 0

    # Quadrant-based exploration - ensure we cover all areas
    exploration_quadrant: int = 0  # 0=NE, 1=SE, 2=SW, 3=NW
    quadrant_start_step: int = 0
    steps_per_quadrant: int = 50  # Explore each quadrant for this many steps (adaptive based on map)
    observed_map_extent: int = 0  # Track how far we've explored to estimate map size

    # Track which resource types have been found (to know if we need to explore more)
    found_resource_types: set[str] = field(default_factory=set)


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

        # Use larger map size to handle big maps like machina_1 (200x200)
        # We start at center and can explore in any direction
        map_size = 500
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

        # Observation-based stuck detection (more reliable than landmark verification)
        # Hash the observation to detect if view has changed
        obs_hash = hash(tuple(
            (tok.location, tok.feature.name, tok.value)
            for tok in obs.tokens
            if tok.feature.name == "tag"  # Only hash object positions
        ))
        if obs_hash == state.prev_obs_hash:
            state.same_observation_count += 1
        else:
            state.same_observation_count = 0
            state.prev_obs_hash = obs_hash
            # Observation changed - reset consecutive_failed_moves since we moved
            state.consecutive_failed_moves = 0
            # Don't exit stuck recovery based on observation alone
            # Only exit when we see actual exploration progress (explored_cells grows)

        # If observation hasn't changed for many steps, we're REALLY stuck
        if state.same_observation_count >= 5:
            state.stuck_recovery_active = True

        # ALWAYS rotate quadrant based on step count for systematic coverage
        # This ensures we explore all directions regardless of stuck recovery status
        if state.step_count - state.quadrant_start_step > state.steps_per_quadrant:
            state.exploration_quadrant = (state.exploration_quadrant + 1) % 4
            state.quadrant_start_step = state.step_count

        # Force exploration of new areas if we have some resources but haven't found extractors for others
        # This handles the case where the agent is stuck in one area of a cave map
        if state.heart_recipe:
            deficits = {
                "carbon": max(0, state.heart_recipe.get("carbon", 0) - state.carbon),
                "oxygen": max(0, state.heart_recipe.get("oxygen", 0) - state.oxygen),
                "germanium": max(0, state.heart_recipe.get("germanium", 0) - state.germanium),
                "silicon": max(0, state.heart_recipe.get("silicon", 0) - state.silicon),
            }
            # Check if we need resources we haven't found extractors for
            missing_without_known_extractors = False
            for resource, deficit in deficits.items():
                if deficit > 0 and resource not in state.found_resource_types:
                    missing_without_known_extractors = True
                    break

            # If we've explored for a while without finding needed extractors, reset position
            # This is more aggressive than the general stuck recovery
            if missing_without_known_extractors and state.step_count > 200:
                # Check if we've been in this quadrant too long without finding new resource types
                steps_in_quadrant = state.step_count - state.quadrant_start_step
                if steps_in_quadrant > state.steps_per_quadrant // 3:  # Rotate faster when missing resources
                    # Force rotation to next quadrant early
                    state.exploration_quadrant = (state.exploration_quadrant + 1) % 4
                    state.quadrant_start_step = state.step_count
                    state.cached_path = None
                    state.cached_path_target = None
                    # Also activate stuck recovery to use simpler observation-based movement
                    # This helps break out of caves where pathfinding keeps failing
                    state.stuck_recovery_active = True

        # Track exploration progress - if explored_cells hasn't grown, we're not making progress
        current_explored = len(state.explored_cells)
        if current_explored > state.last_explored_count:
            state.last_explored_count = current_explored
            state.last_exploration_progress_step = state.step_count
            # Made progress - can exit stuck recovery now
            if state.stuck_recovery_active:
                state.stuck_recovery_active = False
        elif state.step_count - state.last_exploration_progress_step > 50:
            # No exploration progress for 100 steps - activate stuck recovery
            state.stuck_recovery_active = True
            # Also clear stale map data since we might have position drift
            state.cached_path = None
            state.cached_path_target = None
            # Reset position to center and clear explored cells to start fresh
            # Keep station locations (charger/assembler/chest) since they're critical
            if state.step_count - state.last_exploration_progress_step > 200:
                map_center = state.map_height // 2
                state.row = map_center
                state.col = map_center
                state.explored_cells.clear()
                state.extractors = {"carbon": [], "oxygen": [], "germanium": [], "silicon": []}
                state.found_resource_types.clear()
                # Don't clear stations - they're critical for recharge/assembly/delivery
                state.last_exploration_progress_step = state.step_count
                state.last_explored_count = 0  # Reset so we can detect progress after reset
                state.stuck_recovery_active = False  # Allow normal behavior after reset
                # Move to next quadrant to explore new areas after reset
                state.exploration_quadrant = (state.exploration_quadrant + 1) % 4
                state.quadrant_start_step = state.step_count

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
            self._store_landmarks(state, obs)
            return

        # If we were using an object, position doesn't change
        if state.using_object_this_step:
            state.using_object_this_step = False
            self._store_landmarks(state, obs)
            return

        # Get expected movement delta
        direction = state.last_action.name[5:]  # Remove "move_" prefix
        expected_delta = self._move_deltas.get(direction)
        if expected_delta is None:
            self._store_landmarks(state, obs)
            return

        dr, dc = expected_delta

        # Check if move succeeded by verifying landmark position shift
        move_succeeded = self._verify_move_success(state, obs, dr, dc)

        if move_succeeded:
            state.row += dr
            state.col += dc
            # Invalidate cached path since we moved
            state.cached_path = None
            # Reset drift counter on successful move
            state.consecutive_failed_moves = 0
            state.last_successful_move_step = state.step_count
            # Exit stuck recovery mode
            state.stuck_recovery_active = False
        else:
            # Track failed moves for drift detection
            state.consecutive_failed_moves += 1
            # If too many consecutive failed moves, invalidate cached paths
            # This suggests position drift - our map may be out of sync
            if state.consecutive_failed_moves >= 5:
                state.cached_path = None
                state.cached_path_target = None

        # Store new landmark for next step
        self._store_landmarks(state, obs)

    def _verify_move_success(self, state: HarvestState, obs: AgentObservation, dr: int, dc: int) -> bool:
        """Verify if the last move succeeded using multiple landmarks.

        Uses consensus from multiple landmarks. Only confirms move if at least one
        landmark verifies it. Denies move if any landmark contradicts it.
        """
        if not state.prev_landmarks:
            # No previous landmarks - can't verify, be conservative
            return True

        verified_count = 0
        contradicted_count = 0

        for prev_pos, prev_tag in state.prev_landmarks:
            prev_r, prev_c = prev_pos

            # Expected new position of landmark in observation coords
            # Movement is opposite in observation space: if we move north (dr=-1),
            # objects shift south in our view (+1 row)
            expected_new_r = prev_r - dr
            expected_new_c = prev_c - dc

            # Find this landmark in current observation
            for tok in obs.tokens:
                if tok.feature.name == "tag" and tok.value == prev_tag:
                    obs_r, obs_c = tok.location
                    # If landmark is at expected position, move succeeded
                    if obs_r == expected_new_r and obs_c == expected_new_c:
                        verified_count += 1
                        break
                    # If landmark is at same position as before, move failed
                    if obs_r == prev_r and obs_c == prev_c:
                        contradicted_count += 1
                        break

        # If ANY landmark contradicts the move, it failed
        if contradicted_count > 0:
            return False

        # If at least one landmark verified, move succeeded
        if verified_count > 0:
            return True

        # No landmarks could be verified - this can happen when:
        # 1. Moving into open areas with no nearby landmarks
        # 2. All landmarks moved out of view
        # In this case, check if we have ANY objects visible that could serve as landmarks
        # If we see walls/extractors/stations, assume move succeeded (we're in a valid area)
        for tok in obs.tokens:
            if tok.feature.name == "tag":
                tag_name = self._tag_names.get(tok.value, "").lower()
                if any(x in tag_name for x in ["wall", "extractor", "assembler", "chest", "charger"]):
                    return True

        # Truly open area with nothing visible - assume move succeeded
        # (Being too conservative here causes exploration to break)
        return True

    def _store_landmarks(self, state: HarvestState, obs: AgentObservation) -> None:
        """Store multiple stable landmarks from current observation for move verification.

        We store walls and other static objects near the center (so they stay in view after moves).
        """
        landmarks = []
        center_r, center_c = self._obs_hr, self._obs_wr

        for tok in obs.tokens:
            if tok.feature.name != "tag":
                continue

            obs_r, obs_c = tok.location
            # Skip center (that's us)
            if obs_r == center_r and obs_c == center_c:
                continue

            tag_name = self._tag_names.get(tok.value, "").lower()

            # Only use static objects as landmarks
            is_static = any(x in tag_name for x in ["wall", "extractor", "assembler", "chest", "charger"])
            if not is_static:
                continue

            # Prefer landmarks CLOSE to center (so they stay in view after a move)
            dist_from_center = abs(obs_r - center_r) + abs(obs_c - center_c)

            # Only use landmarks within 3 cells of center (guaranteed to stay in view)
            if dist_from_center <= 3:
                landmarks.append(((obs_r, obs_c), tok.value, dist_from_center))

        # Sort by distance (closest first) and keep up to 5 landmarks
        landmarks.sort(key=lambda x: x[2])
        state.prev_landmarks = [(pos, tag) for pos, tag, _ in landmarks[:5]]

    def _discover_objects(self, state: HarvestState, parsed: ParsedObservation) -> None:
        """Update occupancy map and discover extractors/stations."""
        if state.row < 0:
            return

        # Mark all observed cells as FREE initially and track as explored
        center = state.map_height // 2  # Our starting position in internal map
        for obs_r in range(2 * self._obs_hr + 1):
            for obs_c in range(2 * self._obs_wr + 1):
                r = obs_r - self._obs_hr + state.row
                c = obs_c - self._obs_wr + state.col
                if 0 <= r < state.map_height and 0 <= c < state.map_width:
                    state.occupancy[r][c] = CellType.FREE.value
                    state.explored_cells.add((r, c))
                    # Track map extent from starting position
                    extent = max(abs(r - center), abs(c - center))
                    if extent > state.observed_map_extent:
                        state.observed_map_extent = extent
                        # Adapt steps_per_quadrant based on observed map size
                        # Small maps (< 20): 25 steps per quadrant
                        # Medium maps (20-100): 50 steps per quadrant
                        # Large maps (100-150): 150 steps per quadrant
                        # Very large maps (> 150): 250 steps per quadrant
                        if extent < 20:
                            state.steps_per_quadrant = 25
                        elif extent < 100:
                            state.steps_per_quadrant = 50
                        elif extent < 150:
                            state.steps_per_quadrant = 150
                        else:
                            state.steps_per_quadrant = 250

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
        # Track that we've found this resource type
        state.found_resource_types.add(resource_type)

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
            # GATHER: set vibe to target resource for extraction
            # This is required for some extractors that have resource-specific protocols
            target_resource = self._get_target_resource(state)
            if target_resource and target_resource in RESOURCE_VIBES:
                return RESOURCE_VIBES[target_resource]
            return "default"

    def _get_target_resource(self, state: HarvestState) -> str | None:
        """Get the highest priority resource we need to gather."""
        deficits = self._calculate_deficits(state)
        if all(d <= 0 for d in deficits.values()):
            return None

        # Sort resources by deficit - prioritize whichever we need most
        def resource_priority(res):
            deficit = deficits.get(res, 0)
            if deficit <= 0:
                return -1000
            germanium_bonus = 10 if res == "germanium" else 0
            return deficit + germanium_bonus

        priority_order = sorted(
            ["carbon", "oxygen", "germanium", "silicon"],
            key=resource_priority,
            reverse=True
        )
        for resource in priority_order:
            if deficits.get(resource, 0) > 0:
                return resource
        return None

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
        """Gather resources using hybrid observation + map-based navigation.

        Dynamically prioritizes resources based on:
        1. Largest deficit (need more of this)
        2. Germanium gets slight priority (longest cooldown)
        """
        from .utils import find_direction_to_object_from_obs

        # Find which resources we still need
        deficits = self._calculate_deficits(state)
        if all(d <= 0 for d in deficits.values()):
            return self._actions.noop.Noop()

        # Sort resources by deficit - prioritize whichever we need most
        # Don't give location bonus anymore - it causes the agent to keep going back
        # to extractors that might be on cooldown instead of exploring for new ones
        def resource_priority(res):
            deficit = deficits.get(res, 0)
            if deficit <= 0:
                return -1000  # Don't need this
            # Simple priority: just use deficit + small germanium bonus
            # Higher deficit = higher priority (need more of this resource)
            germanium_bonus = 10 if res == "germanium" else 0
            return deficit + germanium_bonus

        priority_order = sorted(
            ["carbon", "oxygen", "germanium", "silicon"],
            key=resource_priority,
            reverse=True
        )

        # Priority 1: Check if any READY extractor is visible and adjacent
        for resource_type in priority_order:
            if deficits.get(resource_type, 0) <= 0:
                continue  # Don't need this resource
            result = self._find_ready_extractor_in_obs(state, resource_type)
            if result is not None:
                obs_direction, _cooldown = result
                state.using_object_this_step = True
                return self._actions.move.Move(obs_direction)

        # Priority 2: Navigate towards any visible extractor we need
        germanium_adjacent_on_cooldown = False
        for resource_type in priority_order:
            if deficits.get(resource_type, 0) <= 0:
                continue

            # Check if adjacent to extractor
            ext_direction = find_direction_to_object_from_obs(
                state.current_obs, resource_type,
                obs_hr=self._obs_hr, obs_wr=self._obs_wr, tag_names=self._tag_names,
            )
            if ext_direction is not None:
                # Adjacent to an extractor that's on cooldown (Priority 1 would have extracted if ready)
                if resource_type == "germanium":
                    # Mark that we're waiting for germanium, but continue checking other resources
                    germanium_adjacent_on_cooldown = True
                    continue  # Try other resources first instead of waiting
                # For other resources on cooldown, move away to let them refresh
                # (Don't get stuck waiting at a single extractor)
                continue

            # Check if we can SEE the extractor (visible but not adjacent)
            nav_direction = self._find_extractor_direction_in_obs(state, resource_type)
            if nav_direction is not None:
                # Only move if that direction is clear
                if self._is_direction_clear_in_obs(state, nav_direction):
                    return self._actions.move.Move(nav_direction)
                # Direction blocked - try to go around by finding a clear direction
                for alt_dir in ["north", "south", "east", "west"]:
                    if alt_dir != nav_direction and self._is_direction_clear_in_obs(state, alt_dir):
                        return self._actions.move.Move(alt_dir)

        # If we're adjacent to germanium on cooldown and found nothing else to do, wait
        if germanium_adjacent_on_cooldown:
            return self._actions.noop.Noop()

        # STUCK RECOVERY: If stuck, skip pathfinding and use exploration
        if state.stuck_recovery_active or state.consecutive_failed_moves >= 5:
            return self._explore(state)

        # Priority 3: Navigate to known extractor using map-based pathfinding
        for resource_type in priority_order:
            if deficits.get(resource_type, 0) <= 0:
                continue
            extractor = self._find_nearest_extractor(state, resource_type)
            if extractor is not None:
                # Use pathfinding to navigate to the extractor
                return self._move_towards(state, extractor.position, reach_adjacent=True)

        # Fallback: explore to find resources
        return self._explore(state)

    def _find_ready_extractor_in_obs(self, state: HarvestState, resource_type: str) -> Optional[tuple[str, int]]:
        """Find a ready extractor of given type adjacent to us in observation.

        Returns (direction, cooldown) if found, or None.
        Checks ACTUAL cooldown and clipped status from observation tokens.
        """
        from .utils import find_direction_to_object_from_obs

        obs_direction = find_direction_to_object_from_obs(
            state.current_obs, resource_type,
            obs_hr=self._obs_hr, obs_wr=self._obs_wr, tag_names=self._tag_names,
        )
        if obs_direction is None:
            return None

        # Get cooldown and clipped status from observation at that position
        dir_offsets = {"north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1)}
        dr, dc = dir_offsets[obs_direction]
        extractor_obs_pos = (self._obs_hr + dr, self._obs_wr + dc)

        cooldown = 0  # Default to 0 if not found (ready)
        clipped = 0  # Default to 0 (not clipped)
        remaining_uses = 999  # Default to high value
        for tok in state.current_obs.tokens:
            if tok.location == extractor_obs_pos:
                if tok.feature.name == "cooldown_remaining":
                    cooldown = tok.value
                elif tok.feature.name == "clipped":
                    clipped = tok.value
                elif tok.feature.name == "remaining_uses":
                    remaining_uses = tok.value

        # Only return if ready (not on cooldown, not clipped, has uses remaining)
        if cooldown == 0 and clipped == 0 and remaining_uses > 0:
            return (obs_direction, cooldown)
        return None

    def _find_extractor_direction_in_obs(self, state: HarvestState, resource_type: str) -> Optional[str]:
        """Find direction towards a visible extractor (not necessarily adjacent).

        Scans entire observation for the extractor and returns direction to move towards it.
        """
        extractor_positions = []

        for tok in state.current_obs.tokens:
            if tok.feature.name == "tag":
                tag_name = self._tag_names.get(tok.value, "").lower()
                if resource_type in tag_name and "extractor" in tag_name:
                    extractor_positions.append(tok.location)

        if not extractor_positions:
            return None

        center = (self._obs_hr, self._obs_wr)
        closest = min(extractor_positions, key=lambda p: abs(p[0] - center[0]) + abs(p[1] - center[1]))

        dr = closest[0] - center[0]
        dc = closest[1] - center[1]

        if abs(dr) >= abs(dc):
            if dr < 0:
                return "north"
            elif dr > 0:
                return "south"
        if dc < 0:
            return "west"
        elif dc > 0:
            return "east"

        return None

    def _find_station_adjacent_in_obs(self, state: HarvestState, station_type: str) -> Optional[str]:
        """Find if a station (assembler/chest/charger) is adjacent in observation.

        Returns direction to the station if adjacent, or None.
        """
        dir_offsets = {"north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1)}

        for direction, (dr, dc) in dir_offsets.items():
            adj_obs_pos = (self._obs_hr + dr, self._obs_wr + dc)
            for tok in state.current_obs.tokens:
                if tok.location == adj_obs_pos and tok.feature.name == "tag":
                    tag_name = self._tag_names.get(tok.value, "").lower()
                    if station_type in tag_name:
                        return direction
        return None

    def _find_station_direction_in_obs(self, state: HarvestState, station_type: str) -> Optional[tuple[str, str | None]]:
        """Find directions towards a visible station (not necessarily adjacent).

        Scans entire observation for the station and returns (primary_dir, secondary_dir).
        Primary is the axis with larger distance, secondary is the other axis.
        """
        station_positions = []

        for tok in state.current_obs.tokens:
            if tok.feature.name == "tag":
                tag_name = self._tag_names.get(tok.value, "").lower()
                if station_type in tag_name:
                    station_positions.append(tok.location)

        if not station_positions:
            return None

        center = (self._obs_hr, self._obs_wr)
        closest = min(station_positions, key=lambda p: abs(p[0] - center[0]) + abs(p[1] - center[1]))

        dr = closest[0] - center[0]
        dc = closest[1] - center[1]

        # Determine primary and secondary directions
        row_dir = "north" if dr < 0 else ("south" if dr > 0 else None)
        col_dir = "west" if dc < 0 else ("east" if dc > 0 else None)

        if abs(dr) >= abs(dc):
            return (row_dir, col_dir) if row_dir else (col_dir, None)
        else:
            return (col_dir, row_dir) if col_dir else (row_dir, None)

    def _navigate_to_station(self, state: HarvestState, station_type: str) -> Action:
        """Navigate to a station using hybrid observation + map-based navigation.

        Strategy:
        0. If stuck, skip pathfinding (but still check observation)
        1. If station is adjacent in observation - use it
        2. If station is visible in observation - navigate to it
        3. If station position is known in memory - use pathfinding
        4. Otherwise explore to find the station
        """
        # Priority 1: Check if station is adjacent in observation - use it
        # (Always check this, even when stuck - we might have found it!)
        adj_dir = self._find_station_adjacent_in_obs(state, station_type)
        if adj_dir is not None:
            state.using_object_this_step = True
            # Exit stuck recovery since we found what we need
            state.stuck_recovery_active = False
            return self._actions.move.Move(adj_dir)

        # Priority 2: Check if station is visible (but not adjacent)
        nav_result = self._find_station_direction_in_obs(state, station_type)
        if nav_result is not None:
            primary_dir, secondary_dir = nav_result
            # Try primary direction first
            if primary_dir and self._is_direction_clear_in_obs(state, primary_dir):
                return self._actions.move.Move(primary_dir)
            # Try secondary direction (moving along the other axis)
            if secondary_dir and self._is_direction_clear_in_obs(state, secondary_dir):
                return self._actions.move.Move(secondary_dir)
            # Both blocked - try any clear direction
            for alt_dir in ["north", "south", "east", "west"]:
                if self._is_direction_clear_in_obs(state, alt_dir):
                    return self._actions.move.Move(alt_dir)

        # STUCK RECOVERY: If stuck, skip pathfinding and use exploration
        # (Pathfinding uses our position tracker which may have drifted)
        if state.stuck_recovery_active or state.consecutive_failed_moves >= 5:
            return self._explore(state)

        # Priority 3: Use pathfinding to known station location
        station_pos = state.stations.get(station_type)
        if station_pos is not None:
            action = self._move_towards(state, station_pos, reach_adjacent=True, station_name=station_type)
            if action.name != "noop":
                return action

        # Station not visible and not known - explore to find it
        return self._explore(state)

    def _do_assemble(self, state: HarvestState) -> Action:
        """Assemble hearts at the assembler."""
        return self._navigate_to_station(state, "assembler")

    def _do_deliver(self, state: HarvestState) -> Action:
        """Deliver hearts to the chest."""
        return self._navigate_to_station(state, "chest")

    def _do_recharge(self, state: HarvestState) -> Action:
        """Recharge at the charger."""
        return self._navigate_to_station(state, "charger")

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
            # Pathfinding failed - just try any clear direction
            # (Don't call _explore here to avoid recursion)
            for direction in ["north", "south", "east", "west"]:
                if self._is_direction_clear_in_obs(state, direction):
                    return self._actions.move.Move(direction)
            return self._actions.noop.Noop()

        # Get next step
        next_pos = path[0]
        state.cached_path = path[1:] if len(path) > 1 else None

        # Convert to action
        dr = next_pos[0] - state.row
        dc = next_pos[1] - state.col

        # Determine direction from delta
        direction = None
        if dr == -1 and dc == 0:
            direction = "north"
        elif dr == 1 and dc == 0:
            direction = "south"
        elif dr == 0 and dc == 1:
            direction = "east"
        elif dr == 0 and dc == -1:
            direction = "west"

        if direction is None:
            return self._actions.noop.Noop()

        # CRITICAL: Validate move is actually clear using observation
        # This prevents position drift from causing us to walk into walls
        if self._is_direction_clear_in_obs(state, direction):
            return self._actions.move.Move(direction)

        # Path says to go this way but observation shows it's blocked
        # Invalidate the cached path and try alternatives
        state.cached_path = None
        state.cached_path_target = None

        # Try other clear directions that make progress towards target
        clear_dirs = [d for d in ["north", "south", "east", "west"]
                      if self._is_direction_clear_in_obs(state, d)]

        if not clear_dirs:
            return self._actions.noop.Noop()

        # Pick direction that moves us closer to target
        target_dr = target[0] - state.row
        target_dc = target[1] - state.col

        # Prioritize directions that reduce distance to target
        def direction_score(d):
            deltas = {"north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1)}
            ddr, ddc = deltas[d]
            # Score is negative distance after move (higher is better)
            new_dr = target_dr - ddr
            new_dc = target_dc - ddc
            return -(abs(new_dr) + abs(new_dc))

        best_dir = max(clear_dirs, key=direction_score)
        return self._actions.move.Move(best_dir)

    def _explore(self, state: HarvestState) -> Action:
        """Explore using frontier-based navigation for efficient map coverage.

        Strategy:
        0. STUCK RECOVERY: If stuck for many steps, cycle through all directions
        1. Check for visible extractors we need (use immediately)
        2. Check for known extractors in memory (navigate via pathfinding)
        3. Use frontier-based exploration to systematically cover the map
        4. Fall back to observation-based movement if no frontiers
        """
        # STUCK RECOVERY: When stuck for 5+ moves, ignore pathfinding and cycle directions
        # This handles position drift where our map-based navigation is unreliable
        # Lower threshold (5 instead of 10) to catch drift earlier
        if state.consecutive_failed_moves >= 5:
            state.stuck_recovery_active = True
            # Clear any cached paths - they're based on drifted position
            state.cached_path = None
            state.cached_path_target = None
            # Clear discovered objects since position is unreliable
            if state.consecutive_failed_moves >= 15:
                state.extractors = {"carbon": [], "oxygen": [], "germanium": [], "silicon": []}
                state.found_resource_types.clear()
                state.stations = {"assembler": None, "chest": None, "charger": None}
                state.explored_cells.clear()
                # Also rotate quadrant to explore new areas
                state.exploration_quadrant = (state.exploration_quadrant + 1) % 4
                state.quadrant_start_step = state.step_count

        if state.stuck_recovery_active:
            # Quadrant-based exploration: systematically explore all areas
            # Quadrants: 0=NE (north+east), 1=SE, 2=SW, 3=NW
            # Rotate quadrant every N steps to ensure full coverage
            if state.step_count - state.quadrant_start_step > state.steps_per_quadrant:
                state.exploration_quadrant = (state.exploration_quadrant + 1) % 4
                state.quadrant_start_step = state.step_count

            # Direction preferences for each quadrant
            quadrant_dirs = {
                0: ["north", "east"],   # NE
                1: ["south", "east"],   # SE
                2: ["south", "west"],   # SW
                3: ["north", "west"],   # NW
            }
            preferred = quadrant_dirs[state.exploration_quadrant]

            # Try preferred directions first
            for direction in preferred:
                if self._is_direction_clear_in_obs(state, direction):
                    return self._actions.move.Move(direction)

            # Preferred blocked - try any clear direction
            all_dirs = ["north", "east", "south", "west"]
            clear_dirs = [d for d in all_dirs if self._is_direction_clear_in_obs(state, d)]
            if clear_dirs:
                # Pick one that's not opposite to our quadrant direction
                opposite = {"north": "south", "south": "north", "east": "west", "west": "east"}
                non_opposite = [d for d in clear_dirs if d not in [opposite.get(p) for p in preferred]]
                if non_opposite:
                    return self._actions.move.Move(random.choice(non_opposite))
                return self._actions.move.Move(random.choice(clear_dirs))

            # All directions blocked - noop
            return self._actions.noop.Noop()

        deficits = self._calculate_deficits(state)

        # Sort resources by deficit (like _do_gather does) - prioritize resources we need most
        def resource_priority(res):
            deficit = deficits.get(res, 0)
            if deficit <= 0:
                return -1000  # Don't need this
            germanium_bonus = 10 if res == "germanium" else 0
            return deficit + germanium_bonus

        priority_order = sorted(
            ["carbon", "oxygen", "germanium", "silicon"],
            key=resource_priority,
            reverse=True
        )

        # Priority 1: Check for READY extractor adjacent to us
        for resource_type in priority_order:
            if deficits.get(resource_type, 0) <= 0:
                continue
            result = self._find_ready_extractor_in_obs(state, resource_type)
            if result is not None:
                obs_direction, _cooldown = result
                state.using_object_this_step = True
                return self._actions.move.Move(obs_direction)

        # Priority 2: Check for visible extractor we need (navigate in obs)
        for resource_type in priority_order:
            if deficits.get(resource_type, 0) <= 0:
                continue
            nav_direction = self._find_extractor_direction_in_obs(state, resource_type)
            if nav_direction is not None:
                if self._is_direction_clear_in_obs(state, nav_direction):
                    return self._actions.move.Move(nav_direction)

        # Priority 3: Navigate to known extractor using map-based pathfinding
        for resource_type in priority_order:
            if deficits.get(resource_type, 0) <= 0:
                continue
            extractor = self._find_nearest_extractor(state, resource_type)
            if extractor is not None:
                # Use pathfinding to navigate to the extractor
                return self._move_towards(state, extractor.position, reach_adjacent=True)

        # Priority 4: Frontier-based exploration - go to unexplored areas
        # Prefer frontiers in the current quadrant direction for systematic coverage
        frontiers = self._find_frontier_targets(state)
        if frontiers:
            current_pos = (state.row, state.col)

            # Get quadrant bias - prefer frontiers in the current quadrant direction
            # Quadrants: 0=NE (north+east), 1=SE, 2=SW, 3=NW
            quadrant_bias = {
                0: (-1, 1),   # NE: prefer north (negative row) and east (positive col)
                1: (1, 1),    # SE: prefer south (positive row) and east (positive col)
                2: (1, -1),   # SW: prefer south and west (negative col)
                3: (-1, -1),  # NW: prefer north and west
            }
            row_bias, col_bias = quadrant_bias[state.exploration_quadrant]

            def frontier_score(f):
                # Distance from current position
                dist = abs(f[0] - current_pos[0]) + abs(f[1] - current_pos[1])
                # Quadrant bonus (large bonus for being in preferred direction)
                row_diff = f[0] - current_pos[0]
                col_diff = f[1] - current_pos[1]
                quadrant_bonus = 0
                if row_bias * row_diff > 0:  # Same sign = preferred direction
                    quadrant_bonus += 200  # Strong bonus for correct quadrant
                if col_bias * col_diff > 0:
                    quadrant_bonus += 200
                # Small distance penalty - we want to explore FURTHER into the quadrant
                # not just nearby frontiers. Cap the penalty so quadrant bonus dominates.
                dist_penalty = min(dist, 50)  # Cap at 50 so quadrant bonus (up to 400) dominates
                return -dist_penalty + quadrant_bonus

            best_frontier = max(frontiers, key=frontier_score)
            # Navigate to frontier using pathfinding
            action = self._move_towards(state, best_frontier, reach_adjacent=False)
            if action.name != "noop":
                return action

        # Fallback: Move using quadrant-based exploration
        # Rotate quadrant every N steps to ensure full coverage
        if state.step_count - state.quadrant_start_step > state.steps_per_quadrant:
            state.exploration_quadrant = (state.exploration_quadrant + 1) % 4
            state.quadrant_start_step = state.step_count

        # Direction preferences for each quadrant
        quadrant_dirs = {
            0: ["north", "east"],   # NE
            1: ["south", "east"],   # SE
            2: ["south", "west"],   # SW
            3: ["north", "west"],   # NW
        }
        preferred = quadrant_dirs[state.exploration_quadrant]

        # Try preferred directions first
        for direction in preferred:
            if self._is_direction_clear_in_obs(state, direction):
                return self._actions.move.Move(direction)

        # Preferred blocked - try any clear direction
        all_dirs = ["north", "east", "south", "west"]
        clear_dirs = [d for d in all_dirs if self._is_direction_clear_in_obs(state, d)]
        if clear_dirs:
            return self._actions.move.Move(random.choice(clear_dirs))

        return self._actions.noop.Noop()

    def _is_direction_clear_in_obs(self, state: HarvestState, direction: str) -> bool:
        """Check if a direction is clear based on observation."""
        dir_offsets = {"north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1)}
        dr, dc = dir_offsets[direction]
        target_obs_pos = (self._obs_hr + dr, self._obs_wr + dc)

        for tok in state.current_obs.tokens:
            if tok.location == target_obs_pos and tok.feature.name == "tag":
                tag_name = self._tag_names.get(tok.value, "").lower()
                if "wall" in tag_name or "extractor" in tag_name or "assembler" in tag_name or "chest" in tag_name or "charger" in tag_name:
                    return False
        return True

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

    short_names = ["zfogg_harvest", "zfogg_harvest_policy"]

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
