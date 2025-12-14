"""
Simple Baseline Scripted Agent - Minimal implementation for ablation studies.

This agent only implements the core functionality needed to:
1. Explore the map to find extractors, assembler, and chest
2. Gather resources from nearest extractors
3. Deposit resources at the assembler
4. Assemble hearts and deliver to chest

No advanced features: no coordination, no caching, no probes, no visit scoring.
Just simple, clean, correct behavior.
"""

from __future__ import annotations

import random
from typing import Callable, Optional

from mettagrid.config.mettagrid_config import CardinalDirection, CardinalDirections
from mettagrid.policy.policy import MultiAgentPolicy, StatefulAgentPolicy, StatefulPolicyImpl
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action
from mettagrid.simulator.interface import AgentObservation

from .pathfinding import compute_goal_cells, shortest_path
from .pathfinding import is_traversable as path_is_traversable
from .pathfinding import is_within_bounds as path_is_within_bounds
from .types import (
    BaselineHyperparameters,
    CellType,
    ExtractorInfo,
    ObjectState,
    ParsedObservation,
    Phase,
    SimpleAgentState,
)
from .utils import (
    change_vibe_action as utils_change_vibe_action,
)
from .utils import (
    is_adjacent,
    is_station,
    is_wall,
    read_inventory_from_obs,
)
from .utils import (
    parse_observation as utils_parse_observation,
)
from .utils import (
    update_agent_position as utils_update_agent_position,
)
from .utils import (
    use_object_at as utils_use_object_at,
)

# Sentinel for agent-centric features
AGENT_SENTINEL = 0x55


class BaselineAgentPolicyImpl(StatefulPolicyImpl[SimpleAgentState]):
    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        agent_id: int,
        hyperparams: BaselineHyperparameters,
    ):
        self._agent_id = agent_id
        self._hyperparams = hyperparams
        self._policy_env_info = policy_env_info

        # Observation grid half-ranges from config
        self._obs_hr = policy_env_info.obs_height // 2  # Egocentric observation half-radius (rows)
        self._obs_wr = policy_env_info.obs_width // 2  # Egocentric observation half-radius (cols)

        # Action lookup
        self._actions = policy_env_info.actions
        self._move_deltas = {
            "north": (-1, 0),
            "south": (1, 0),
            "east": (0, 1),
            "west": (0, -1),
        }

        # Fast lookup tables for observation feature decoding
        self._spatial_feature_names = {
            "tag",
            "cooldown_remaining",
            "clipped",
            "remaining_uses",
        }
        agent_feature_pairs = {
            "agent:group": "agent_group",
            "agent:frozen": "agent_frozen",
        }
        self._agent_feature_key_by_name: dict[str, str] = agent_feature_pairs

        # Protocol feature prefixes (for dynamic recipe discovery)
        self._protocol_input_prefix = "protocol_input:"
        self._protocol_output_prefix = "protocol_output:"

        # Map resource names to their corresponding vibe names for debugging glyphs.
        # This keeps resource naming (carbon, oxygen, germanium, silicon) separate from
        # visual glyph naming (carbon_a, oxygen_a, etc.).
        self._resource_to_vibe: dict[str, str] = {
            "carbon": "carbon_a",
            "oxygen": "oxygen_a",
            "germanium": "germanium_a",
            "silicon": "silicon_a",
        }

    def initial_agent_state(self) -> SimpleAgentState:
        """Get initial state for an agent."""
        # Cache tag name mapping for efficient tag -> object name lookup
        self._tag_names = self._policy_env_info.tag_id_to_name

        # Use a reasonable default size for origin-relative positioning
        # The agent will expand the map dynamically as it explores
        map_size = 200  # Large enough for most missions
        center = map_size // 2  # Agent starts at center of this larger map

        # Initialize heart recipe from protocols passed via PolicyEnvInterface
        heart_recipe = None
        for protocol in self._policy_env_info.assembler_protocols:
            if protocol.output_resources.get("heart", 0) > 0:
                # Use this protocol's input resources as the heart recipe
                heart_recipe = dict(protocol.input_resources)
                # Remove energy from the recipe if present (agents don't track energy as a gatherable resource)
                heart_recipe.pop("energy", None)
                break

        return SimpleAgentState(
            agent_id=self._agent_id,
            map_height=map_size,
            map_width=map_size,
            occupancy=[[CellType.FREE.value] * map_size for _ in range(map_size)],
            row=center,
            col=center,
            heart_recipe=heart_recipe,
        )

    def step_with_state(self, obs: AgentObservation, s: SimpleAgentState) -> tuple[Action, SimpleAgentState]:
        """Main step function that processes observation and returns action with updated state."""
        s.step_count += 1

        # Store observation for collision detection
        s.current_obs = obs
        s.agent_occupancy.clear()

        # Read inventory from observation
        read_inventory_from_obs(s, obs, obs_hr=self._obs_hr, obs_wr=self._obs_wr)

        # Update agent position based on last action
        self._update_agent_position(s)

        # Parse observation to get structured data
        parsed = self.parse_observation(s, obs)

        # Update occupancy map and discover objects
        self._update_occupancy_and_discover(s, parsed)

        # Update phase based on current state
        self._update_phase(s)

        # Update vibe to match phase
        desired_vibe = self._get_vibe_for_phase(s.phase, s)
        if s.current_glyph != desired_vibe:
            s.current_glyph = desired_vibe
            # Return vibe change action this step
            action = utils_change_vibe_action(desired_vibe, actions=self._actions)
            s.last_action = action
            return action, s

        # Check for stuck state and handle escape
        stuck_action = self._check_stuck_and_escape(s)
        if stuck_action is not None:
            s.last_action = stuck_action
            return stuck_action, s

        # Execute action for current phase
        action = self._execute_phase(s)

        # Save action for next step's position update
        s.last_action = action

        return action, s

    def parse_observation(
        self, state: SimpleAgentState, obs: AgentObservation, debug: bool = False
    ) -> ParsedObservation:
        """Parse token-based observation into structured format."""
        return utils_parse_observation(
            state,
            obs,
            obs_hr=self._obs_hr,
            obs_wr=self._obs_wr,
            spatial_feature_names=self._spatial_feature_names,
            agent_feature_key_by_name=self._agent_feature_key_by_name,
            protocol_input_prefix=self._protocol_input_prefix,
            protocol_output_prefix=self._protocol_output_prefix,
            tag_names=self._tag_names,
            debug=debug,
        )

    def _try_random_direction(self, s: SimpleAgentState) -> Optional[Action]:
        """Try to move in any free adjacent direction, avoiding agent collisions. Returns None if all blocked."""
        directions: list[CardinalDirection] = ["north", "south", "east", "west"]
        random.shuffle(directions)
        for direction in directions:
            dr, dc = self._move_deltas[direction]
            nr, nc = s.row + dr, s.col + dc

            # Check if cell is traversable
            if not (path_is_within_bounds(s, nr, nc) and s.occupancy[nr][nc] == CellType.FREE.value):
                continue

            # Check for agent collision at target world coordinates
            if (nr, nc) in s.agent_occupancy:
                continue

            return self._actions.move.Move(direction)
        return None

    def _clear_stuck_state(self, s: SimpleAgentState) -> None:
        """Clear all stuck detection state."""
        s.stuck_loop_detected = False

    def _check_stuck_and_escape(self, s: SimpleAgentState) -> Optional[Action]:
        """Check if agent is stuck in a loop and return escape action if needed."""
        if not self._hyperparams.stuck_detection_enabled or not s.stuck_loop_detected:
            return None

        action = self._try_random_direction(s)
        self._clear_stuck_state(s)
        return action if action else self._actions.noop.Noop()

    def _update_occupancy_and_discover(self, s: SimpleAgentState, parsed: ParsedObservation) -> None:
        """Update occupancy map and discover objects from parsed observation."""
        # Discover heart recipe from assembler protocol (if not yet discovered)
        if s.heart_recipe is None:
            for _pos, obj_state in parsed.nearby_objects.items():
                if obj_state.name == "assembler" and obj_state.protocol_inputs:
                    # Check if this is the heart recipe (outputs "heart")
                    if obj_state.protocol_outputs.get("heart", 0) > 0:
                        s.heart_recipe = dict(obj_state.protocol_inputs)
                        break

        # Update occupancy map and discover extractors/stations
        self._discover_objects(s, parsed)

    def _update_agent_position(self, s: SimpleAgentState) -> None:
        """Update agent position based on last action.

        Position is tracked relative to origin (starting position), using only movement deltas.
        No dependency on simulation.grid_objects().

        IMPORTANT: When using objects (extractors, stations), the agent "moves into" them but doesn't
        actually change position. We detect this by checking the using_object_this_step flag.
        """
        # Use utility function for basic position update
        utils_update_agent_position(s, move_deltas=self._move_deltas)

        # Update position history and detect loops
        current_pos = (s.row, s.col)
        s.position_history.append(current_pos)
        if len(s.position_history) > self._hyperparams.position_history_size:
            s.position_history.pop(0)

        # Detect if agent is stuck in a back-and-forth loop
        # Check if last 4-6 positions show oscillation (e.g., A->B->A->B or A->B->C->A->B->C)
        if len(s.position_history) >= 6:
            # Check for 2-position loop (A->B->A->B->A->B)
            if (
                s.position_history[-1] == s.position_history[-3] == s.position_history[-5]
                and s.position_history[-2] == s.position_history[-4] == s.position_history[-6]
                and s.position_history[-1] != s.position_history[-2]
            ):
                s.stuck_loop_detected = True
                s.stuck_escape_step = s.step_count
            # Check for 3-position loop (A->B->C->A->B->C)
            elif len(s.position_history) >= 9:
                if (
                    s.position_history[-1] == s.position_history[-4] == s.position_history[-7]
                    and s.position_history[-2] == s.position_history[-5] == s.position_history[-8]
                    and s.position_history[-3] == s.position_history[-6] == s.position_history[-9]
                ):
                    s.stuck_loop_detected = True
                    s.stuck_escape_step = s.step_count

    def _discover_objects(self, s: SimpleAgentState, parsed: ParsedObservation) -> None:
        """Discover extractors and stations from parsed observation."""
        if s.row < 0:
            return

        # First pass: Mark ALL observed cells as FREE (will be updated to OBSTACLE below if needed)
        # This ensures empty cells are marked as traversable
        for obs_r in range(2 * self._obs_hr + 1):
            for obs_c in range(2 * self._obs_wr + 1):
                # Convert observation-relative coords to world coords
                r, c = obs_r - self._obs_hr + s.row, obs_c - self._obs_wr + s.col
                if 0 <= r < s.map_height and 0 <= c < s.map_width:
                    s.occupancy[r][c] = CellType.FREE.value

        # Second pass: mark obstacles and discover objects
        for pos, obj_state in parsed.nearby_objects.items():
            r, c = pos
            obj_name = obj_state.name.lower()

            # Walls are obstacles
            if is_wall(obj_name):
                s.occupancy[r][c] = CellType.OBSTACLE.value
                continue

            # Other agents: track their positions but don't mark as obstacles
            if obj_name == "agent" and obj_state.agent_id != s.agent_id:
                s.agent_occupancy.add((r, c))
                continue

            # Discover stations (all stations are obstacles - can't walk through them)
            for station_name in ("assembler", "chest", "charger"):
                if is_station(obj_name, station_name):
                    s.occupancy[r][c] = CellType.OBSTACLE.value
                    self._discover_station(s, pos, station_name)
                    break
            else:
                # Extractors are also obstacles
                if "extractor" in obj_name:
                    s.occupancy[r][c] = CellType.OBSTACLE.value
                    resource_type = obj_name.replace("_extractor", "").replace("clipped_", "")
                    if resource_type:
                        self._discover_extractor(s, pos, resource_type, obj_state)

    def _discover_station(self, s: SimpleAgentState, pos: tuple[int, int], station_key: str) -> None:
        """Record a discovered station location if not already known."""
        if s.stations.get(station_key) is None:
            s.stations[station_key] = pos

    def _discover_extractor(
        self,
        s: SimpleAgentState,
        pos: tuple[int, int],
        resource_type: str,
        obj_state: ObjectState,
    ) -> None:
        extractor = None
        for existing in s.extractors[resource_type]:
            if existing.position == pos:
                extractor = existing
                break

        if extractor is None:
            extractor = ExtractorInfo(
                position=pos,
                resource_type=resource_type,
                last_seen_step=s.step_count,
            )
            s.extractors[resource_type].append(extractor)

        extractor.last_seen_step = s.step_count
        extractor.cooldown_remaining = obj_state.cooldown_remaining
        extractor.clipped = obj_state.clipped > 0
        extractor.remaining_uses = obj_state.remaining_uses

    def _update_phase(self, s: SimpleAgentState) -> None:
        """
        Update agent phase based on current goals (no arbitrary thresholds).

        Priority order:
        1. RECHARGE if energy low
        2. DELIVER if have hearts
        3. ASSEMBLE if have all 4 resources
        4. GATHER (default) - collect resources, explore if needed
        """
        old_phase = s.phase

        # Priority 1: Recharge if energy low
        # Enter RECHARGE if energy drops below threshold
        if s.energy < self._hyperparams.recharge_threshold_low:
            if s.phase != Phase.RECHARGE:
                s.phase = Phase.RECHARGE
                # Clear extractor waiting state when leaving GATHER
                s.pending_use_resource = None
                s.pending_use_amount = 0
                s.waiting_at_extractor = None
            return

        # Stay in RECHARGE until energy is fully restored
        if s.phase == Phase.RECHARGE:
            if s.energy >= self._hyperparams.recharge_threshold_high:
                s.phase = Phase.GATHER
                s.target_position = None
            # Still recharging, stay in this phase
            return

        # Priority 2: Deliver hearts if we have any
        if s.hearts > 0:
            if s.phase != Phase.DELIVER:
                s.phase = Phase.DELIVER
                # Clear extractor waiting state when leaving GATHER
                s.pending_use_resource = None
                s.pending_use_amount = 0
                s.waiting_at_extractor = None
            return

        # Priority 3: Assemble if we have all resources
        # Only check if we've discovered the recipe
        can_assemble = False
        if s.heart_recipe is not None:
            can_assemble = (
                s.carbon >= s.heart_recipe.get("carbon", 0)
                and s.oxygen >= s.heart_recipe.get("oxygen", 0)
                and s.germanium >= s.heart_recipe.get("germanium", 0)
                and s.silicon >= s.heart_recipe.get("silicon", 0)
            )

        if can_assemble:
            if s.phase != Phase.ASSEMBLE:
                s.phase = Phase.ASSEMBLE
                # Clear extractor waiting state when leaving GATHER
                s.pending_use_resource = None
                s.pending_use_amount = 0
                s.waiting_at_extractor = None
            return

        # Priority 4: Default to GATHER
        # GATHER will explore internally when it can't find needed extractors
        if s.phase != Phase.GATHER:
            s.phase = Phase.GATHER
            s.target_position = None
            s.pending_use_resource = None
            s.pending_use_amount = 0
            s.waiting_at_extractor = None

        # Invalidate cached path if phase changed (likely targeting something different now)
        if old_phase != s.phase:
            s.cached_path = None
            s.cached_path_target = None

    def _get_vibe_for_phase(self, phase: Phase, state: SimpleAgentState) -> str:
        """Map phase to a vibe for visual debugging in replays."""
        # During GATHER, vibe the target resource we're currently collecting
        if phase == Phase.GATHER and state.target_resource is not None:
            # Map resource name (e.g., "silicon") to a valid vibe name (e.g., "silicon_a").
            return self._resource_to_vibe.get(state.target_resource, "default")

        phase_to_vibe = {
            Phase.GATHER: "carbon_a",  # Default fallback if no target resource
            Phase.ASSEMBLE: "heart_a",  # Red for assembly
            Phase.DELIVER: "default",  # Must be "default" to deposit hearts into chest
            Phase.RECHARGE: "charger",  # Blue/electric for recharging
            Phase.CRAFT_UNCLIP: "gear",  # Gear icon for crafting unclip items
            Phase.UNCLIP: "gear",  # Gear icon for unclipping
        }
        return phase_to_vibe.get(phase, "default")

    def _calculate_deficits(self, s: SimpleAgentState) -> dict[str, int]:
        """Calculate how many more of each resource we need for a heart."""
        # Recipe must be discovered from observations - no hardcoded fallback
        if s.heart_recipe is None:
            raise RuntimeError(
                "Heart recipe not discovered! Agent must observe assembler with correct vibe to learn recipe. "
                "Ensure protocol_details_obs=True in game config."
            )

        return {
            "carbon": max(0, s.heart_recipe.get("carbon", 0) - s.carbon),
            "oxygen": max(0, s.heart_recipe.get("oxygen", 0) - s.oxygen),
            "germanium": max(0, s.heart_recipe.get("germanium", 0) - s.germanium),
            "silicon": max(0, s.heart_recipe.get("silicon", 0) - s.silicon),
        }

    def _execute_phase(self, s: SimpleAgentState) -> Action:
        """Execute action for current phase."""
        if s.phase == Phase.GATHER:
            return self._do_gather(s)
        elif s.phase == Phase.ASSEMBLE:
            return self._do_assemble(s)
        elif s.phase == Phase.DELIVER:
            return self._do_deliver(s)
        elif s.phase == Phase.RECHARGE:
            return self._do_recharge(s)
        elif s.phase == Phase.UNCLIP:
            return self._do_unclip(s)
        return self._actions.noop.Noop()

    def _explore_directional(self, s: SimpleAgentState) -> Action:
        """
        Simple directional exploration: pick a random direction and stick to it for ~15 steps.
        Changes direction when blocked or after persistence expires.
        Uses move_towards for actual movement to benefit from collision detection and pathfinding checks.

        Anti-stuck mechanism: If stuck in a small area (configurable via hyperparameters) for
        a configurable number of steps, navigates to assembler for a configurable duration
        to escape, then continues exploring.
        """
        if s.row < 0:
            return self._actions.noop.Noop()

        # Check if we're in escape mode (navigating to assembler)
        if s.exploration_escape_until_step > 0:
            if s.step_count >= s.exploration_escape_until_step:
                # Done escaping, continue normal exploration
                s.exploration_escape_until_step = 0
            else:
                # Still in escape mode - navigate to assembler
                if s.stations["assembler"] is not None:
                    # Check if we've reached the assembler (adjacent)
                    if is_adjacent((s.row, s.col), s.stations["assembler"]):
                        # Reached assembler! Exit escape mode
                        s.exploration_escape_until_step = 0
                    else:
                        return self._move_towards(s, s.stations["assembler"], reach_adjacent=True)
                else:
                    # Don't know where assembler is yet, just continue exploring
                    s.exploration_escape_until_step = 0

        # Check if stuck in small area using last N positions from history
        if len(s.position_history) >= self._hyperparams.exploration_area_check_window:
            recent_positions = s.position_history[-self._hyperparams.exploration_area_check_window :]
            min_row = min(pos[0] for pos in recent_positions)
            max_row = max(pos[0] for pos in recent_positions)
            min_col = min(pos[1] for pos in recent_positions)
            max_col = max(pos[1] for pos in recent_positions)

            area_height = max_row - min_row + 1
            area_width = max_col - min_col + 1

            if (
                area_height <= self._hyperparams.exploration_area_size_threshold
                and area_width <= self._hyperparams.exploration_area_size_threshold
                and s.stations["assembler"] is not None
            ):
                assembler_pos = s.stations["assembler"]
                if assembler_pos is not None:
                    dist = abs(s.row - assembler_pos[0]) + abs(s.col - assembler_pos[1])
                    # Only trigger escape if far from assembler and not already in escape mode
                    # Also prevent triggering too frequently (wait at least 25 steps since last escape ended)
                    if (
                        dist > self._hyperparams.exploration_assembler_distance_threshold
                        and s.exploration_escape_until_step == 0
                    ):
                        # Stuck in small area and far from assembler! Enter escape mode
                        s.exploration_escape_until_step = s.step_count + self._hyperparams.exploration_escape_duration
                        return self._move_towards(s, assembler_pos, reach_adjacent=True)
        # Check if we should keep current exploration direction
        if s.exploration_target_step is not None:
            steps_in_direction = s.step_count - s.exploration_target_step
            if steps_in_direction < self._hyperparams.exploration_direction_persistence:
                # Still committed to current direction, try to move that way
                if s.exploration_target is not None and isinstance(s.exploration_target, str):
                    # exploration_target stores the direction as a string
                    direction = s.exploration_target
                    # Calculate target position in that direction
                    dr, dc = {"north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1)}.get(direction, (0, 0))
                    next_r, next_c = s.row + dr, s.col + dc

                    # Use move_towards to handle the movement (includes all checks)
                    action = self._move_towards(s, (next_r, next_c))

                    # If move_towards returned noop (blocked), pick a new direction
                    if action == self._actions.noop.Noop():
                        # Fall through to pick new direction
                        pass
                    else:
                        return action

        # Need to pick a new direction
        # Try all directions and pick a random valid one
        valid_directions = []
        for direction in CardinalDirections:
            dr, dc = {"north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1)}.get(direction, (0, 0))
            next_r, next_c = s.row + dr, s.col + dc
            if path_is_traversable(s, next_r, next_c, CellType):
                valid_directions.append(direction)

        if valid_directions:
            # Pick a random valid direction
            new_direction = random.choice(valid_directions)
            s.exploration_target = new_direction  # Store direction as string
            s.exploration_target_step = s.step_count

            # Calculate target position for the new direction
            dr, dc = {"north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1)}.get(new_direction, (0, 0))
            next_r, next_c = s.row + dr, s.col + dc

            # Use move_towards for actual movement
            return self._move_towards(s, (next_r, next_c))

        # No valid moves, use try_random_direction (which also checks agent collisions)
        random_action = self._try_random_direction(s)
        return random_action if random_action else self._actions.noop.Noop()

    def _explore_until(
        self, s: SimpleAgentState, condition: Callable[[], bool], reason: str = "Exploring"
    ) -> Action | None:
        """
        Explore until a condition is met.

        Args:
            condition: Function that returns True when exploration should stop
            reason: Description for logging

        Returns:
            Exploration action if condition not met, None if condition met
        """
        if condition():
            # Condition met, stop exploring
            return None

        # For goal-directed exploration, disable escape mode
        # so we can focus on finding the target object (extractor, assembler, etc.)
        # Escape mode will resume during pure exploration if needed
        s.exploration_escape_until_step = 0

        return self._explore(s)

    def _explore(self, s: SimpleAgentState) -> Action:
        """Execute exploration using directional strategy."""
        return self._explore_directional(s)

    def _find_any_needed_extractor(self, s: SimpleAgentState) -> tuple[ExtractorInfo, str] | None:
        """
        Find ANY extractor that produces ANY resource we need.
        Prioritizes resources by deficit size (biggest deficit first).
        Returns (extractor, resource_type) tuple, or None if nothing found.
        """
        deficits = self._calculate_deficits(s)
        # Sort resources by deficit size (largest first)
        sorted_resources = sorted(deficits.items(), key=lambda x: x[1], reverse=True)

        # Check resources in order of deficit size
        for resource_type, deficit in sorted_resources:
            if deficit > 0:
                extractor = self._find_nearest_extractor(s, resource_type)
                if extractor is not None:
                    return (extractor, resource_type)

        return None

    def _find_extractor_at_position(self, s: SimpleAgentState, pos: tuple[int, int]) -> Optional[ExtractorInfo]:
        """Find extractor at given position in agent's known extractors."""
        for extractor_list in s.extractors.values():
            for ext in extractor_list:
                if ext.position == pos:
                    return ext
        return None

    def _clear_waiting_state(self, s: SimpleAgentState) -> None:
        """Clear extractor waiting state."""
        s.waiting_at_extractor = None
        s.wait_steps = 0

    def _handle_waiting_for_extractor(self, s: SimpleAgentState) -> Optional[Action]:
        """Handle waiting for activated extractor. Returns Noop if still waiting, None if done."""
        if s.pending_use_resource is None or s.waiting_at_extractor is None:
            return None

        # Check if we received resources (inventory increased)
        current_amount = getattr(s, s.pending_use_resource, 0)
        if current_amount > s.pending_use_amount:
            # Success - clear pending state and continue gathering
            s.pending_use_resource = None
            s.pending_use_amount = 0
            self._clear_waiting_state(s)
            return None  # Continue with next resource

        # Look up the extractor we're waiting for
        extractor = self._find_extractor_at_position(s, s.waiting_at_extractor)

        # Calculate timeout based on observed cooldown
        max_wait = extractor.cooldown_remaining + 5 if extractor else 20

        s.wait_steps += 1
        if s.wait_steps > max_wait:
            # Timeout - reset and try again
            s.pending_use_resource = None
            s.pending_use_amount = 0
            self._clear_waiting_state(s)
            return None  # Let _do_gather try again immediately

        return self._actions.noop.Noop()

    def _navigate_to_adjacent(
        self, s: SimpleAgentState, target_pos: tuple[int, int], target_name: str = "target"
    ) -> Optional[Action]:
        """Navigate to adjacent cell of target. Returns action if navigating, None if already adjacent.

        This is a generic helper used for both extractors and stations.
        """
        is_adjacent_to_target = is_adjacent((s.row, s.col), target_pos)

        if is_adjacent_to_target:
            return None  # Already adjacent

        # Move towards target
        action = self._move_towards(s, target_pos, reach_adjacent=True)
        if action == self._actions.noop.Noop():
            return self._explore(s)
        return action

    def _use_extractor_if_ready(self, s: SimpleAgentState, extractor: ExtractorInfo, resource_type: str) -> Action:
        """Try to use extractor if ready. Returns appropriate action."""

        # Wait if on cooldown
        if extractor.cooldown_remaining > 0:
            s.waiting_at_extractor = extractor.position
            s.wait_steps += 1
            return self._actions.noop.Noop()

        # Skip if depleted/clipped
        if extractor.remaining_uses == 0 or extractor.clipped:
            self._clear_waiting_state(s)
            return self._actions.noop.Noop()

        # Use it! Track pre-use inventory and activate
        old_amount = getattr(s, resource_type, 0)

        # Set waiting state to detect when resource is received
        s.pending_use_resource = resource_type
        s.pending_use_amount = old_amount
        s.waiting_at_extractor = extractor.position

        return utils_use_object_at(
            s,
            extractor.position,
            actions=self._actions,
            move_deltas=self._move_deltas,
            using_for=f"{resource_type}_extractor",
        )

    def _do_gather(self, s: SimpleAgentState) -> Action:
        """
        Gather resources from nearest extractors.
        Opportunistically uses ANY extractor for ANY needed resource.
        """
        # Handle waiting for activated extractor
        wait_action = self._handle_waiting_for_extractor(s)
        if wait_action is not None:
            return wait_action

        # Check resource deficits
        deficits = self._calculate_deficits(s)

        if all(d <= 0 for d in deficits.values()):
            self._clear_waiting_state(s)
            return self._actions.noop.Noop()

        # Explore until we find an extractor for a needed resource

        explore_action = self._explore_until(
            s,
            condition=lambda: self._find_any_needed_extractor(s) is not None,
            reason=f"Need extractors for: {', '.join(k for k, v in deficits.items() if v > 0)}",
        )
        if explore_action is not None:
            return explore_action

        # Found an extractor - navigate and use it

        result = self._find_any_needed_extractor(s)
        if result is None:
            return self._explore(s)  # Shouldn't happen, but be safe

        extractor, resource_type = result
        s.exploration_target = None  # Clear exploration target
        s.target_resource = resource_type

        # Navigate to extractor if not adjacent
        nav_action = self._navigate_to_adjacent(s, extractor.position, target_name=f"{resource_type}_extractor")
        if nav_action is not None:
            self._clear_waiting_state(s)
            return nav_action

        # Adjacent - try to use it
        return self._use_extractor_if_ready(s, extractor, resource_type)

    def _do_assemble(self, s: SimpleAgentState) -> Action:
        """Assemble hearts at assembler."""
        # Explore until we find assembler
        explore_action = self._explore_until(
            s, condition=lambda: s.stations["assembler"] is not None, reason="Need assembler"
        )
        if explore_action is not None:
            return explore_action

        # Assembler is known, navigate to it and use it
        assembler = s.stations["assembler"]
        assert assembler is not None  # Guaranteed by _explore_until above

        # Navigate to adjacent cell
        nav_action = self._navigate_to_adjacent(s, assembler, target_name="assembler")
        if nav_action is not None:
            return nav_action

        # Adjacent - use it
        return utils_use_object_at(
            s, assembler, actions=self._actions, move_deltas=self._move_deltas, using_for="assembler"
        )

    def _do_deliver(self, s: SimpleAgentState) -> Action:
        """Deliver hearts to chest."""
        # Explore until we find chest
        explore_action = self._explore_until(s, condition=lambda: s.stations["chest"] is not None, reason="Need chest")
        if explore_action is not None:
            return explore_action

        # Chest is known, navigate to it and use it
        chest = s.stations["chest"]
        assert chest is not None  # Guaranteed by _explore_until above

        # Navigate to adjacent cell
        nav_action = self._navigate_to_adjacent(s, chest, target_name="chest")
        if nav_action is not None:
            return nav_action

        # Adjacent - use it
        return utils_use_object_at(s, chest, actions=self._actions, move_deltas=self._move_deltas, using_for="chest")

    def _do_recharge(self, s: SimpleAgentState) -> Action:
        """Recharge at charger."""
        # Explore until we find charger
        explore_action = self._explore_until(
            s, condition=lambda: s.stations["charger"] is not None, reason="Need charger"
        )
        if explore_action is not None:
            return explore_action

        # Charger is known, navigate to it and use it
        charger = s.stations["charger"]
        assert charger is not None  # Guaranteed by _explore_until above

        # Navigate to adjacent cell
        nav_action = self._navigate_to_adjacent(s, charger, target_name="charger")
        if nav_action is not None:
            return nav_action

        # Adjacent - use it
        return utils_use_object_at(
            s, charger, actions=self._actions, move_deltas=self._move_deltas, using_for="charger"
        )

    def _do_unclip(self, s: SimpleAgentState) -> Action:
        """Unclip extractors - this is implemented in the UnclippingAgent."""
        s.phase = Phase.GATHER
        return self._actions.noop.Noop()

    def _find_nearest_extractor(self, s: SimpleAgentState, resource_type: str) -> Optional[ExtractorInfo]:
        """Find the nearest AVAILABLE extractor of the given type."""
        extractors = s.extractors.get(resource_type, [])
        if not extractors:
            return None

        # Filter out clipped or depleted extractors
        available = [e for e in extractors if not e.clipped and e.remaining_uses > 0]

        if not available:
            return None

        def distance(pos: tuple[int, int]) -> int:
            return abs(pos[0] - s.row) + abs(pos[1] - s.col)

        return min(available, key=lambda e: distance(e.position))

    def _move_towards(
        self,
        s: SimpleAgentState,
        target: tuple[int, int],
        *,
        reach_adjacent: bool = False,
        allow_goal_block: bool = False,
    ) -> Action:
        """Pathfind toward a target using BFS with obstacle awareness.

        Uses path caching to avoid recomputing BFS every step - only recomputes when:
        - Target changed
        - Path parameters changed (reach_adjacent)
        - Next step is blocked
        - Path completed or invalid
        """

        start = (s.row, s.col)
        if start == target and not reach_adjacent:
            return self._actions.noop.Noop()

        goal_cells = compute_goal_cells(s, target, reach_adjacent, CellType)
        if not goal_cells:
            return self._actions.noop.Noop()

        # Check if we can reuse cached path
        path = None
        if (
            s.cached_path
            and len(s.cached_path) > 0
            and s.cached_path_target == target
            and s.cached_path_reach_adjacent == reach_adjacent
        ):
            # Validate cached path - check if next step is still walkable
            next_pos = s.cached_path[0]
            if path_is_traversable(s, next_pos[0], next_pos[1], CellType) or (
                allow_goal_block and next_pos in goal_cells
            ):
                # Cached path is valid! Use it
                path = s.cached_path
            # If next step blocked, fall through to recompute

        # Need to recompute path
        if path is None:
            path = shortest_path(s, start, goal_cells, allow_goal_block, CellType)
            # Cache the new path
            s.cached_path = path.copy() if path else None
            s.cached_path_target = target
            s.cached_path_reach_adjacent = reach_adjacent
        if not path:
            # No path found - try a random direction to escape
            random_action = self._try_random_direction(s)
            return random_action if random_action else self._actions.noop.Noop()

        # Get next step from path
        next_pos = path[0]

        # Advance cached path (remove the step we're about to take)
        if s.cached_path and len(s.cached_path) > 0:
            s.cached_path = s.cached_path[1:]  # Remove first element
            # Clear cache if we've reached the end
            if len(s.cached_path) == 0:
                s.cached_path = None
                s.cached_path_target = None

        # Convert next position to action
        dr = next_pos[0] - s.row
        dc = next_pos[1] - s.col

        # Check if there's an agent at the target location in observations
        # Calculate observation coordinates for the target cell
        target_obs_r = self._obs_hr + dr
        target_obs_c = self._obs_wr + dc

        if dr == -1 and dc == 0:
            action = "north"
        elif dr == 1 and dc == 0:
            action = "south"
        elif dr == 0 and dc == 1:
            action = "east"
        elif dr == 0 and dc == -1:
            action = "west"
        else:
            return self._actions.noop.Noop()

        # Check for agent collision and try alternative direction if blocked
        if self._is_agent_at_obs_location(s, target_obs_r, target_obs_c):
            # Agent collision detected! Try a random collision-free direction
            random_action = self._try_random_direction(s)
            if random_action:
                s.cached_path = None
                s.cached_path_target = None
                return random_action
            # No valid moves, just noop
            return self._actions.noop.Noop()

        return self._actions.move.Move(action)

    def _is_agent_at_obs_location(self, s: SimpleAgentState, obs_r: int, obs_c: int) -> bool:
        """Check if there's an agent at the given observation coordinates.

        Args:
            s: Agent state
            obs_r: Row in observation space (relative to agent's view)
            obs_c: Column in observation space (relative to agent's view)

        Returns:
            True if an agent is detected at that location
        """
        if s.current_obs is None:
            return False

        for tok in s.current_obs.tokens:
            if tok.location == (obs_r, obs_c):
                # Check for agent:group feature which indicates another agent
                if tok.feature.name == "agent:group":
                    return True
        return False

    def _move_into_cell(self, s: SimpleAgentState, target: tuple[int, int]) -> Action:
        """Return the action that attempts to step into the target cell.

        Checks for agent occupancy before moving to avoid collisions.
        """
        tr, tc = target
        if s.row == tr and s.col == tc:
            return self._actions.noop.Noop()
        dr = tr - s.row
        dc = tc - s.col

        # Check if another agent is at the target position
        if (tr, tc) in s.agent_occupancy:
            # Another agent is blocking the target, wait or try alternative
            random_action = self._try_random_direction(s)
            return random_action if random_action else self._actions.noop.Noop()

        if dr == -1:
            return self._actions.move.Move("north")
        if dr == 1:
            return self._actions.move.Move("south")
        if dc == 1:
            return self._actions.move.Move("east")
        if dc == -1:
            return self._actions.move.Move("west")
        # Fallback to pathfinder if offsets unexpected
        return self._move_towards(s, target, allow_goal_block=True)


# ============================================================================
# Policy Wrapper Classes
# ============================================================================


class BaselinePolicy(MultiAgentPolicy):
    short_names = ["scripted_baseline"]

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        device: str = "cpu",
        hyperparams: Optional[BaselineHyperparameters] = None,
    ):
        super().__init__(policy_env_info, device=device)
        self._agent_policies: dict[int, StatefulAgentPolicy[SimpleAgentState]] = {}
        self._hyperparams = hyperparams or BaselineHyperparameters()

    def agent_policy(self, agent_id: int) -> StatefulAgentPolicy[SimpleAgentState]:
        if agent_id not in self._agent_policies:
            self._agent_policies[agent_id] = StatefulAgentPolicy(
                BaselineAgentPolicyImpl(self._policy_env_info, agent_id, self._hyperparams),
                self._policy_env_info,
                agent_id=agent_id,
            )
        return self._agent_policies[agent_id]


RESOURCE_VIBE_ALIASES: dict[str, str] = {
    "carbon": "carbon_a",
    "oxygen": "oxygen_a",
    "germanium": "germanium_a",
    "silicon": "silicon_a",
    # Crafting resources (appear when crafting unclipping items)
    "decoder": "gear",
    "modulator": "gear",
    "resonator": "gear",
    "scrambler": "gear",
}
