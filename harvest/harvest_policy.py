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

import logging
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

# Import refactored managers
from .exploration import ExplorationManager
from .energy import EnergyManager
from .resources import ResourceManager
from .navigation import NavigationManager
from .state_tracker import StateTracker
from .map import MapManager
from .maze_navigation import MazeNavigator, WallFollowMode
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
class ChargerInfo:
    """Track quality/reliability of a charger."""
    position: tuple[int, int]
    times_approached: int = 0
    times_successfully_used: int = 0
    last_attempt_step: int = 0

    @property
    def success_rate(self) -> float:
        """Success rate for this charger (0.0 to 1.0)."""
        if self.times_approached == 0:
            return 1.0  # Optimistic for new chargers
        return self.times_successfully_used / self.times_approached

    @property
    def is_reliable(self) -> bool:
        """True if this charger has proven reliable."""
        # Require at least 50% success rate, or give benefit of doubt if < 3 attempts
        return self.success_rate > 0.5 or self.times_approached < 3


@dataclass
class MissionProfile:
    """Detected characteristics of the current mission."""
    map_size: str  # "small" (<30), "medium" (30-100), "large" (>100)
    energy_difficulty: str = "normal"  # "normal" or "starved"

    # Adaptive thresholds based on mission profile
    @property
    def recharge_critical(self) -> int:
        """Critical energy threshold - MUST recharge immediately."""
        return 10

    @property
    def recharge_low(self) -> int:
        """Low energy threshold - enter RECHARGE phase."""
        if self.map_size == "small":
            return 20  # More aggressive on small maps
        elif self.map_size == "medium":
            return 30
        else:  # large
            return 35  # Conservative on large maps

    @property
    def recharge_high(self) -> int:
        """High energy threshold - stay in RECHARGE until this level."""
        if self.map_size == "small":
            return 85  # Don't waste time waiting for full charge
        elif self.map_size == "medium":
            return 75
        else:  # large
            return 60  # Exit recharge sooner to continue exploration

    @property
    def stuck_threshold(self) -> int:
        """Steps stuck before triggering recovery."""
        return 5 if self.map_size == "small" else 10


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

    # CRITICAL: Track ALL discovered chargers for intelligent energy management
    # This allows the agent to use the nearest charger instead of always going to the first one found
    discovered_chargers: list[tuple[int, int]] = field(default_factory=list)
    found_initial_charger: bool = False  # True once we've found at least one charger

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
    exploration_resume_position: Optional[tuple[int, int]] = None  # Where to resume exploring after recharge

    # OSCILLATION PREVENTION: Track committed direction and target
    committed_exploration_direction: Optional[str] = None  # Direction we're committed to (for momentum)
    committed_direction_steps: int = 0  # How many steps we've been going this direction
    committed_frontier_target: Optional[tuple[int, int]] = None  # Frontier target we're committed to
    frontier_target_commitment_steps: int = 0  # Steps remaining with current frontier target

    # Current observation
    current_obs: Optional[AgentObservation] = None

    # Pathfinding cache (initialized after map_size is known)
    path_cache: Optional[PathCache] = None

    # Move verification - store multiple landmarks for robust verification
    prev_landmarks: list[tuple[tuple[int, int], int]] = field(default_factory=list)  # [(obs_pos, tag_id), ...]
    prev_energy: Optional[int] = None  # Previous step's energy for move verification

    # Drift detection - track consecutive failed moves
    consecutive_failed_moves: int = 0
    last_successful_move_step: int = 0

    # Single-use tracking - remember which extractors we've exhausted
    used_extractors: set[tuple[int, int]] = field(default_factory=set)  # Positions of depleted extractors

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

    # MISSION-AWARE ARCHITECTURE: Adapt strategy based on mission characteristics
    mission_profile: Optional[MissionProfile] = None

    # CHARGER QUALITY TRACKING: Remember which chargers work and which don't
    charger_info: dict[tuple[int, int], ChargerInfo] = field(default_factory=dict)
    current_recharge_target: Optional[tuple[int, int]] = None  # Which charger we're currently trying to reach

    # DEAD-END AVOIDANCE: Remember positions where we got stuck with nothing useful
    # Once explored and found empty, mark as dead-end and never return
    dead_end_positions: set[tuple[int, int]] = field(default_factory=set)

    # POSITION-BASED STUCK DETECTION: Track recent positions to detect being stuck at a location
    position_history: list[tuple[int, int]] = field(default_factory=list)  # Last 10 positions
    forced_exploration_direction: Optional[str] = None  # Force exploration in this direction when badly stuck


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
        self._recharge_critical = 10  # STOP ALL non-recharge activity
        self._recharge_low = 35  # Enter dedicated recharge phase (enough to reach charger)
        self._recharge_high = 95  # Charge to 95+ (chargers give +14, so 84→98 is typical)

        # Set up detailed file logging to debug policy behavior
        self._logger = logging.getLogger(f"HarvestPolicy.Agent{agent_id}")
        self._logger.setLevel(logging.DEBUG)
        # Clear any existing handlers to avoid duplicates
        self._logger.handlers.clear()
        # Create file handler
        fh = logging.FileHandler("harvest.log", mode='w')  # Overwrite each run
        fh.setLevel(logging.DEBUG)
        # Create formatter with timestamp, level, and message
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        fh.setFormatter(formatter)
        self._logger.addHandler(fh)
        self._logger.info(f"=== HarvestPolicy Agent {agent_id} initialized ===")

        # Initialize managers (refactored architecture)
        self._tag_names = policy_env_info.tags  # Get tag names for observation parsing
        self.exploration = ExplorationManager(self._obs_hr, self._obs_wr, self._tag_names)
        self.energy = EnergyManager()
        self.resources = ResourceManager(self._logger)
        self.navigation = NavigationManager(self._logger)
        self.state_tracker = StateTracker(self._obs_hr, self._obs_wr, self._tag_names, self._logger)
        self.map_manager: Optional[MapManager] = None  # Initialized on first step (needs map dimensions)
        self.maze_navigator: Optional[MazeNavigator] = None  # Initialized in initial_agent_state() after tag_id_to_name is available

    def initial_agent_state(self) -> HarvestState:
        """Initialize state for the agent."""
        self._tag_names = self._policy_env_info.tag_id_to_name

        # Initialize MazeNavigator now that we have the correct tag_id_to_name dict
        self.maze_navigator = MazeNavigator(self._logger, self._obs_hr, self._obs_wr, self._tag_names)

        # Initialize heart recipe from assembler protocols
        # Try to find a valid recipe that makes sense
        heart_recipe = None
        for protocol in self._policy_env_info.assembler_protocols:
            if protocol.output_resources.get("heart", 0) > 0:
                recipe = dict(protocol.input_resources)
                recipe.pop("energy", None)  # Energy is not a gatherable resource

                # Validate recipe: must require at least one resource we can extract
                has_valid_resources = False
                for resource in ["carbon", "oxygen", "germanium", "silicon"]:
                    if recipe.get(resource, 0) > 0:
                        has_valid_resources = True
                        break

                if has_valid_resources:
                    heart_recipe = recipe
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

        # Initialize MapManager on first step (needs map dimensions from state)
        if self.map_manager is None:
            self.map_manager = MapManager(
                state.map_height,
                state.map_width,
                self._obs_hr,
                self._obs_wr,
                self._tag_names,
                self._logger
            )
            self._logger.info(f"  MAP: Initialized {state.map_height}x{state.map_width} map")

        # Update inventory from observation
        read_inventory_from_obs(state, obs, obs_hr=self._obs_hr, obs_wr=self._obs_wr)

        # Update map from current observation
        self.map_manager.update_from_observation(state)

        # MISSION DETECTION: Detect mission profile early to adapt strategy
        # Do detection after a few steps to get map extent, but not too late
        if state.mission_profile is None and state.step_count >= 5:
            # Wait a few steps to get better map size estimate
            state.mission_profile = self._detect_mission_profile(state)
            self._logger.info(f"  ★ DETECTED MISSION: map_size={state.mission_profile.map_size}, "
                            f"extent={state.observed_map_extent}, "
                            f"recharge_thresholds=({state.mission_profile.recharge_low},{state.mission_profile.recharge_high})")

        # Log current state at start of step
        self._logger.debug(f"Step {state.step_count}: pos=({state.row},{state.col}) energy={state.energy} "
                          f"phase={state.phase.value} inv=(C:{state.carbon} O:{state.oxygen} "
                          f"G:{state.germanium} Si:{state.silicon} H:{state.hearts})")

        # Update position based on last action - but verify move success first!
        self._update_position_with_verification(state, obs)

        # POSITION-BASED STUCK DETECTION (critical fix for energy_starved)
        # Problem: Observation hash changes when energy/inventory changes, resetting consecutive_failed_moves
        # Solution: Track position history - if agent hasn't moved in N steps, it's stuck
        current_pos = (state.row, state.col)
        state.position_history.append(current_pos)
        if len(state.position_history) > 10:
            state.position_history.pop(0)  # Keep last 10 positions

        # Count how many recent positions are the same as current position
        same_position_count = sum(1 for pos in state.position_history if pos == current_pos)

        # If stuck at same position for 5+ steps, set consecutive_failed_moves to match
        # This properly detects stuck state even when energy/inventory changes
        if same_position_count >= 5:
            # Agent is stuck at this position - use max to avoid overwriting higher counts
            state.consecutive_failed_moves = max(state.consecutive_failed_moves, same_position_count)
            self._logger.debug(f"  STUCK DETECTED: same position for {same_position_count} steps (consecutive_fails={state.consecutive_failed_moves})")

        else:
            # Agent is moving - reset counter
            state.consecutive_failed_moves = 0

        # Rotate quadrant based on progress, not just time
        # Only rotate if we're not making exploration progress AND time limit exceeded
        # This ensures we don't abandon a productive quadrant prematurely
        steps_in_quadrant = state.step_count - state.quadrant_start_step
        no_recent_progress = state.step_count - state.last_exploration_progress_step > state.steps_per_quadrant // 2

        # CRITICAL FIX for quadrant_buildings: Force quadrant rotation to ensure coverage
        # Reduce time per quadrant for better coverage on quadrant-specific missions
        max_steps_per_quadrant = min(state.steps_per_quadrant, 200)  # Cap at 200 steps/quadrant
        if steps_in_quadrant > max_steps_per_quadrant and no_recent_progress:
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
        elif state.step_count - state.last_exploration_progress_step > 100:
            # No exploration progress for 100 steps - activate stuck recovery
            state.stuck_recovery_active = True
            # Invalidate only cached paths, NOT the map or position
            # This lets us try frontier-based or exploration-based navigation instead
            state.cached_path = None
            state.cached_path_target = None

        # EMERGENCY FALLBACK: Reset position only in extreme cases
        # With energy-based verification, position drift should be rare
        # Only trigger this if stuck for a VERY long time with MANY failed moves
        # Increased threshold from 300 to 500 steps since energy verification reduces drift
        if (state.step_count - state.last_exploration_progress_step > 500 and
            state.consecutive_failed_moves >= 20):
            # Nuclear option: position might be corrupted despite energy verification
            # This should rarely trigger with the new verification system
            # Reset position but keep stations since they're critical anchors
            map_center = state.map_height // 2
            state.row = map_center
            state.col = map_center
            # Clear extractors but keep station locations
            state.explored_cells.clear()
            state.extractors = {"carbon": [], "oxygen": [], "germanium": [], "silicon": []}
            state.found_resource_types.clear()
            state.last_exploration_progress_step = state.step_count
            state.last_explored_count = 0
            state.consecutive_failed_moves = 0
            state.stuck_recovery_active = False
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
            self._logger.info(f"  VIBE CHANGE: {state.current_glyph} → {desired_vibe}")
            action = change_vibe_action(desired_vibe, actions=self._actions)
            state.last_action = action
            self._logger.debug(f"  ACTION: {action.name}")
            return action, state

        # Execute current phase
        action = self._execute_phase(state)
        state.last_action = action
        self._logger.debug(f"  ACTION: {action.name}")
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
        # If last action wasn't a move, just clear flags and store current energy
        if not state.last_action or not state.last_action.name.startswith("move_"):
            state.using_object_this_step = False
            self._store_landmarks(state, obs)
            state.prev_energy = state.energy  # Store for next step's verification
            return

        # If we were using an object, position doesn't change
        if state.using_object_this_step:
            state.using_object_this_step = False
            self._store_landmarks(state, obs)
            state.prev_energy = state.energy  # Store for next step's verification
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
            self._logger.debug(f"  ✓ Move {direction} SUCCEEDED → new pos ({state.row},{state.col})")
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
            target_r, target_c = state.row + dr, state.col + dc
            self._logger.debug(f"  ✗ Move {direction} FAILED (stayed at ({state.row},{state.col}), "
                             f"tried to go to ({target_r},{target_c})) - "
                             f"consecutive_fails={state.consecutive_failed_moves}")

            # CRITICAL FIX: Mark the target cell as OBSTACLE to learn from failures
            # This prevents repeatedly trying invalid moves to the same cell
            if path_is_within_bounds(state, target_r, target_c):
                state.occupancy[target_r][target_c] = CellType.OBSTACLE.value
                # Remove from explored cells since it's blocked
                state.explored_cells.discard((target_r, target_c))

            # If too many consecutive failed moves, invalidate cached paths
            # This suggests position drift - our map may be out of sync
            if state.consecutive_failed_moves >= 5:
                state.cached_path = None
                state.cached_path_target = None

        # Store new landmark and energy for next step's verification
        self._store_landmarks(state, obs)
        state.prev_energy = state.energy

    def _verify_move_success(self, state: HarvestState, obs: AgentObservation, dr: int, dc: int) -> bool:
        """Verify if the last move succeeded using multiple methods.

        Uses consensus from multiple landmarks. Only confirms move if at least one
        landmark verifies it. Denies move if any landmark contradicts it.

        CRITICAL FIX: When we move north (dr=-1), landmarks move DOWN in our view (expected_r increases).
        This is because our movement and their apparent movement are in same direction.
        If we move north, the world shifts south relative to us, so landmark_row increases.
        """
        # Method 1: Energy-based verification (most reliable when NOT on chargers!)
        # Moving costs 1 energy. But if on a charger, energy can increase.
        if state.prev_energy is not None:
            expected_energy = state.prev_energy - 1
            if state.energy == state.prev_energy:
                # Energy didn't change → move failed (hit wall or invalid)
                return False
            elif state.energy == expected_energy:
                # Energy decreased by exactly 1 → move succeeded
                return True
            elif state.energy > state.prev_energy:
                # Energy INCREASED (from charger) → move probably succeeded
                # Can't verify with energy, rely on landmark verification
                pass
            # Note: If energy decreased but not by exactly 1, something else happened
            # (e.g., energy drain). Continue to landmark verification.

        # Method 2: Landmark-based verification
        if not state.prev_landmarks:
            # No previous landmarks - can't verify with this method
            # If we had energy verification above, we already handled it
            # Otherwise, be conservative (assume move succeeded to avoid over-correction)
            return True

        verified_count = 0
        contradicted_count = 0

        for prev_pos, prev_tag in state.prev_landmarks:
            prev_r, prev_c = prev_pos

            # Expected new position of landmark in observation coords
            # When we move north (dr=-1), landmarks appear to move SOUTH in our view (row increases by +1)
            # When we move east (dc=+1), landmarks appear to move WEST in our view (col decreases by -1)
            # So: expected_new_pos = prev_pos - movement_delta (opposite direction)
            expected_new_r = prev_r - dr
            expected_new_c = prev_c - dc

            # Find this landmark in current observation
            found = False
            for tok in obs.tokens:
                if tok.feature.name == "tag" and tok.value == prev_tag:
                    obs_r, obs_c = tok.location
                    # If landmark is at expected position, move succeeded
                    if obs_r == expected_new_r and obs_c == expected_new_c:
                        verified_count += 1
                        found = True
                        break
                    # If landmark is at same position as before, move failed
                    elif obs_r == prev_r and obs_c == prev_c:
                        contradicted_count += 1
                        found = True
                        break

        # If ANY landmark contradicts the move, it failed
        if contradicted_count > 0:
            return False

        # If at least one landmark verified, move succeeded
        if verified_count > 0:
            return True

        # No landmarks could be verified - this can happen when:
        # 1. Moving into open areas with no nearby landmarks (all moved out of view)
        # 2. All landmarks are ambiguous
        # Be more conservative: if we can't verify with landmarks, check observation quality
        visible_static_objects = 0
        for tok in obs.tokens:
            if tok.feature.name == "tag":
                tag_name = self._tag_names.get(tok.value, "").lower()
                if any(x in tag_name for x in ["wall", "extractor", "assembler", "chest", "charger"]):
                    visible_static_objects += 1

        # Only assume success if we see a decent number of stable objects
        # This prevents false positives in truly empty areas
        if visible_static_objects >= 2:
            return True

        # Sparse environment - can't verify, assume FAILURE to prevent position drift
        # Better to incorrectly reject a valid move than corrupt position tracking
        return False

    def _store_landmarks(self, state: HarvestState, obs: AgentObservation) -> None:
        """Store multiple stable landmarks from current observation for move verification.

        We store walls and other static objects near the center (so they stay in view after moves).
        Uses a larger pool of landmarks for robustness on large maps with sparse visible objects.
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

            # Accept landmarks up to 5 cells away for better coverage on sparse maps
            # (Previous limit of 3 was too restrictive)
            if dist_from_center <= 5:
                landmarks.append(((obs_r, obs_c), tok.value, dist_from_center))

        # Sort by distance (closest first) and keep up to 15 landmarks (was 5)
        # More landmarks = more robust verification on large maps with sparse environments
        landmarks.sort(key=lambda x: x[2])
        state.prev_landmarks = [(pos, tag) for pos, tag, _ in landmarks[:15]]

    def _discover_objects(self, state: HarvestState, parsed: ParsedObservation) -> None:
        """Update occupancy map and discover extractors/stations."""
        if state.row < 0:
            return

        # Mark all observed cells as FREE (but preserve learned obstacles!)
        # CRITICAL: Don't overwrite cells we've learned are obstacles from failed moves
        center = state.map_height // 2  # Our starting position in internal map
        for obs_r in range(2 * self._obs_hr + 1):
            for obs_c in range(2 * self._obs_wr + 1):
                r = obs_r - self._obs_hr + state.row
                c = obs_c - self._obs_wr + state.col
                if 0 <= r < state.map_height and 0 <= c < state.map_width:
                    # Only mark as FREE if we haven't learned it's an obstacle
                    if state.occupancy[r][c] != CellType.OBSTACLE.value:
                        state.occupancy[r][c] = CellType.FREE.value
                    state.explored_cells.add((r, c))
                    # Track map extent from starting position
                    extent = max(abs(r - center), abs(c - center))
                    if extent > state.observed_map_extent:
                        state.observed_map_extent = extent
                        # Adapt steps_per_quadrant based on observed map size
                        # Small maps (< 20): 25 steps per quadrant
                        # Medium maps (20-100): 50 steps per quadrant
                        # Large maps (100-150): 100 steps per quadrant
                        # Very large maps (> 150): 150 steps per quadrant
                        if extent < 20:
                            state.steps_per_quadrant = 25
                        elif extent < 100:
                            state.steps_per_quadrant = 50
                        elif extent < 150:
                            state.steps_per_quadrant = 100
                        else:
                            state.steps_per_quadrant = 150

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

                    # Special handling for chargers: track ALL of them
                    if station_name == "charger":
                        if pos not in state.discovered_chargers:
                            state.discovered_chargers.append(pos)
                            is_first = not state.found_initial_charger
                            state.found_initial_charger = True

                            # CHARGER QUALITY TRACKING: Create ChargerInfo for new chargers
                            if pos not in state.charger_info:
                                state.charger_info[pos] = ChargerInfo(position=pos)
                                self._logger.debug(f"  CHARGER: Created ChargerInfo for charger at {pos}")

                            if is_first:
                                self._logger.info(f"  ★ FOUND INITIAL CHARGER at {pos}! Total chargers: {len(state.discovered_chargers)}")
                            else:
                                self._logger.debug(f"  Found charger #{len(state.discovered_chargers)} at {pos}")

                    # Also store in stations dict (first one found of each type)
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

        # CRITICAL FIX for single_use_swarm: Mark depleted extractors
        if obj_state.remaining_uses == 0:
            state.used_extractors.add(pos)

    def _update_phase(self, state: HarvestState) -> None:
        """Update phase based on current state.

        CRITICAL: Prioritize finding charger before gathering resources.
        This ensures energy safety on energy-starved missions.
        """
        old_phase = state.phase

        # Get adaptive thresholds based on mission profile
        recharge_critical = self._get_recharge_critical(state)
        recharge_low = self._get_recharge_low(state)
        recharge_high = self._get_recharge_high(state)

        # Priority 1: CRITICAL energy - MUST recharge immediately
        if state.energy < recharge_critical:
            # Save current position for exploration resume
            if old_phase == HarvestPhase.GATHER and state.exploration_resume_position is None:
                state.exploration_resume_position = (state.row, state.col)
                self._logger.info(f"  PHASE: Saved exploration resume position {state.exploration_resume_position}")

            state.phase = HarvestPhase.RECHARGE
            if old_phase != state.phase:
                self._logger.info(f"  PHASE: {old_phase.value} → RECHARGE (CRITICAL energy={state.energy} < {recharge_critical})")
            return

        # Priority 2: Low energy - start recharging
        if state.energy < recharge_low:
            # Save current position for exploration resume
            if old_phase == HarvestPhase.GATHER and state.exploration_resume_position is None:
                state.exploration_resume_position = (state.row, state.col)
                self._logger.info(f"  PHASE: Saved exploration resume position {state.exploration_resume_position}")

            state.phase = HarvestPhase.RECHARGE
            if old_phase != state.phase:
                self._logger.info(f"  PHASE: {old_phase.value} → RECHARGE (low energy={state.energy} < {recharge_low})")
            return

        # Stay in recharge until energy well-restored
        # BUT: If stuck trying to recharge and energy is still safe ON LARGE MAPS, exit RECHARGE to explore
        if state.phase == HarvestPhase.RECHARGE and state.energy < recharge_high:
            # Only use aggressive stuck recovery on large maps where chargers may be inaccessible
            is_large_map = state.mission_profile and state.mission_profile.map_size == "large"
            stuck_threshold = state.mission_profile.stuck_threshold if state.mission_profile else 5
            if is_large_map and state.consecutive_failed_moves >= stuck_threshold and state.energy > 20:
                self._logger.info(f"  PHASE: RECHARGE → GATHER (large map, stuck {state.consecutive_failed_moves} steps, energy={state.energy} still safe, will explore)")
                state.phase = HarvestPhase.GATHER
                return
            self._logger.debug(f"  PHASE: Staying in RECHARGE (energy={state.energy} < {recharge_high})")
            return

        # Priority 3: Deliver hearts
        if state.hearts > 0:
            state.phase = HarvestPhase.DELIVER
            if old_phase != state.phase:
                self._logger.info(f"  PHASE: {old_phase.value} → DELIVER (hearts={state.hearts})")
            return

        # Priority 4: Assemble if we have all resources
        can_assemble = self._can_assemble(state)
        if state.heart_recipe:
            self._logger.debug(f"Step {state.step_count}: PHASE CHECK: recipe={state.heart_recipe}, inv=(C:{state.carbon} O:{state.oxygen} Ge:{state.germanium} Si:{state.silicon}), can_assemble={can_assemble}")

        if state.heart_recipe and can_assemble:
            state.phase = HarvestPhase.ASSEMBLE
            if old_phase != state.phase:
                self._logger.info(f"Step {state.step_count}: PHASE: {old_phase.value} → ASSEMBLE (have all resources: C:{state.carbon} O:{state.oxygen} Ge:{state.germanium} Si:{state.silicon})")
            return

        # Priority 5: FIND CHARGER FIRST before gathering
        # This is critical for energy-starved missions
        # Once we have at least one charger, we can safely gather resources
        if not state.found_initial_charger:
            state.phase = HarvestPhase.GATHER  # Use GATHER phase for exploration
            if old_phase != state.phase:
                self._logger.info(f"  PHASE: {old_phase.value} → GATHER (searching for INITIAL CHARGER)")
            else:
                self._logger.debug(f"  PHASE: In GATHER, still searching for initial charger (found={len(state.discovered_chargers)} chargers)")
            return

        # Priority 6: FIND ALL RESOURCE TYPES before gathering
        # Check if we've discovered at least one extractor of each type
        all_types_found = (
            len(self.map_manager.carbon_extractors) > 0 and
            len(self.map_manager.oxygen_extractors) > 0 and
            len(self.map_manager.germanium_extractors) > 0 and
            len(self.map_manager.silicon_extractors) > 0
        )

        if not all_types_found:
            state.phase = HarvestPhase.GATHER  # Use GATHER phase for exploration
            if old_phase != state.phase:
                c_count = len(self.map_manager.carbon_extractors)
                o_count = len(self.map_manager.oxygen_extractors)
                g_count = len(self.map_manager.germanium_extractors)
                s_count = len(self.map_manager.silicon_extractors)
                self._logger.info(f"  PHASE: {old_phase.value} → GATHER (exploring to find all resources: C={c_count} O={o_count} G={g_count} Si={s_count})")
            else:
                if state.step_count % 100 == 0:  # Log every 100 steps to avoid spam
                    c_count = len(self.map_manager.carbon_extractors)
                    o_count = len(self.map_manager.oxygen_extractors)
                    g_count = len(self.map_manager.germanium_extractors)
                    s_count = len(self.map_manager.silicon_extractors)
                    self._logger.debug(f"Step {state.step_count}: PHASE: Still exploring for all resources: C={c_count} O={o_count} G={g_count} Si={s_count}")
            return

        # Priority 7: Gather resources (only after charger found AND all resource types discovered)
        state.phase = HarvestPhase.GATHER
        if old_phase != state.phase:
            self._logger.info(f"  PHASE: {old_phase.value} → GATHER (all resource types found, starting collection)")

    def _detect_mission_profile(self, state: HarvestState) -> MissionProfile:
        """Detect mission characteristics to adapt strategy."""
        # Use observed map extent to estimate map size
        # observed_map_extent is the max distance we've explored from starting position
        map_dimension = state.observed_map_extent * 2  # Estimate full map size

        if map_dimension < 30:
            map_size = "small"
        elif map_dimension < 100:
            map_size = "medium"
        else:
            map_size = "large"

        # TODO: Detect energy difficulty by checking charger availability
        # For now, assume "normal" energy difficulty
        energy_difficulty = "normal"

        return MissionProfile(map_size=map_size, energy_difficulty=energy_difficulty)

    def _get_recharge_critical(self, state: HarvestState) -> int:
        """Get critical energy threshold (adaptive based on mission)."""
        if state.mission_profile:
            return state.mission_profile.recharge_critical
        return self._recharge_critical

    def _get_recharge_low(self, state: HarvestState) -> int:
        """Get low energy threshold (adaptive based on mission)."""
        if state.mission_profile:
            return state.mission_profile.recharge_low
        return self._recharge_low

    def _get_recharge_high(self, state: HarvestState) -> int:
        """Get high energy threshold (adaptive based on mission)."""
        if state.mission_profile:
            return state.mission_profile.recharge_high
        return self._recharge_high

    def _select_best_charger(self, state: HarvestState) -> tuple[int, int]:
        """Select best charger based on reliability and distance.

        Strategy:
        1. If stuck badly (10+ failures), use FARTHEST charger (escape stuck pattern)
        2. If stuck moderately (5-9 failures), try different charger
        3. Otherwise, prefer RELIABLE + NEAREST charger
        """
        if not state.discovered_chargers:
            # Fallback: use nearest charger (shouldn't happen, but be safe)
            return self._find_nearest_charger(state)

        stuck_threshold = state.mission_profile.stuck_threshold if state.mission_profile else 5

        # Strategy 1: Badly stuck - use FARTHEST charger to escape stuck pattern
        if state.consecutive_failed_moves >= stuck_threshold * 2 and len(state.discovered_chargers) > 1:
            chargers_by_distance = sorted(
                state.discovered_chargers,
                key=lambda pos: abs(pos[0] - state.row) + abs(pos[1] - state.col),
                reverse=True  # Farthest first
            )
            target = chargers_by_distance[0]
            self._logger.info(f"  CHARGER: BADLY STUCK ({state.consecutive_failed_moves} steps), using FARTHEST charger at {target}")
            return target

        # Strategy 2: Moderately stuck - try reliable chargers that aren't the current target
        if state.consecutive_failed_moves >= stuck_threshold and len(state.discovered_chargers) > 1:
            # Find reliable chargers (excluding current stuck target if we have one)
            reliable_chargers = [
                pos for pos in state.discovered_chargers
                if pos in state.charger_info and state.charger_info[pos].is_reliable
                and pos != state.current_recharge_target
            ]

            if reliable_chargers:
                # Pick nearest reliable charger (that's not our stuck target)
                target = min(
                    reliable_chargers,
                    key=lambda pos: abs(pos[0] - state.row) + abs(pos[1] - state.col)
                )
                self._logger.info(f"  CHARGER: Stuck {state.consecutive_failed_moves} steps, trying alternate RELIABLE charger at {target}")
                return target
            else:
                # No reliable alternatives, try any different charger
                alt_chargers = [pos for pos in state.discovered_chargers if pos != state.current_recharge_target]
                if alt_chargers:
                    target = min(alt_chargers, key=lambda pos: abs(pos[0] - state.row) + abs(pos[1] - state.col))
                    self._logger.info(f"  CHARGER: Stuck {state.consecutive_failed_moves} steps, trying alternate charger at {target}")
                    return target

        # Strategy 3: Not stuck - prefer RELIABLE + NEAREST charger
        # Score chargers by: reliability (higher is better) and distance (lower is better)
        def score_charger(pos: tuple[int, int]) -> tuple[float, int]:
            """Return (reliability, distance) for sorting (higher reliability, lower distance is better)."""
            info = state.charger_info.get(pos)
            reliability = info.success_rate if info else 1.0  # Optimistic for unknown chargers
            distance = abs(pos[0] - state.row) + abs(pos[1] - state.col)
            return (-reliability, distance)  # Negative reliability so higher values sort first

        target = min(state.discovered_chargers, key=score_charger)
        info = state.charger_info.get(target)
        reliability_str = f"{info.success_rate:.2f}" if info else "unknown"
        self._logger.debug(f"  CHARGER: Selected nearest reliable charger at {target} (reliability={reliability_str})")
        return target

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
            vibe = "heart_a"
        elif state.phase == HarvestPhase.DELIVER:
            vibe = "heart_b"  # Deposit hearts to chest (heart_b transfers +1 heart)
        elif state.phase == HarvestPhase.RECHARGE:
            vibe = "charger"
        else:
            # GATHER phase
            # CRITICAL: If searching for initial charger OR energy is low,
            # set vibe to "charger" so we can use it!
            recharge_low = self._get_recharge_low(state)
            if not state.found_initial_charger or state.energy < recharge_low:
                vibe = "charger"
                if not state.found_initial_charger:
                    self._logger.debug(f"  VIBE: charger (searching for initial charger)")
                else:
                    self._logger.debug(f"  VIBE: charger (energy {state.energy} < {recharge_low})")
            else:
                # Otherwise, set vibe to target resource for extraction
                target_resource = self._get_target_resource(state)
                if target_resource and target_resource in RESOURCE_VIBES:
                    vibe = RESOURCE_VIBES[target_resource]
                else:
                    vibe = "default"

        return vibe

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
        """Gather resources using OBSERVATION-ONLY navigation.

        TEMPORARY: All pathfinding disabled to debug wall collision issues.
        Uses only visible objects in current observation.
        """
        from .utils import find_direction_to_object_from_obs

        # PRIORITY -1: STUCK RECOVERY - Use navigation manager
        if self.navigation.is_stuck(state, threshold=5):
            # Mark dead-end using exploration manager
            num_marked = self.exploration.mark_dead_end(state)
            self._logger.warning(f"  GATHER: Marked {num_marked} dead-end positions")

            # Get recovery direction from navigation manager
            nearest_charger = self._find_nearest_charger(state) if state.discovered_chargers else None
            direction = self.navigation.handle_stuck_recovery(state, nearest_charger)

            # CRITICAL: Check if direction is actually clear before moving!
            if self._is_direction_clear_in_obs(state, direction):
                return self._actions.move.Move(direction)

            # Stuck recovery direction is blocked - try ANY clear direction
            for alt_dir in ["north", "south", "east", "west"]:
                if self._is_direction_clear_in_obs(state, alt_dir):
                    self._logger.warning(f"  GATHER: Stuck recovery blocked, trying {alt_dir}")
                    return self._actions.move.Move(alt_dir)

            # Completely surrounded - noop
            self._logger.error(f"  GATHER: Completely surrounded by walls - using noop")
            return self._actions.noop.Noop()

        # PRIORITY 0: Resume exploration from saved position after recharge
        if state.exploration_resume_position is not None:
            resume_r, resume_c = state.exploration_resume_position
            dist = abs(resume_r - state.row) + abs(resume_c - state.col)

            # If we're close to resume position (within 3 cells), clear it and continue
            if dist <= 3:
                self._logger.info(f"  GATHER: Reached exploration resume position {state.exploration_resume_position}, resuming normal exploration")
                state.exploration_resume_position = None
            else:
                # Navigate back to resume position
                self._logger.debug(f"  GATHER: Navigating to exploration resume position {state.exploration_resume_position} (dist={dist})")
                direction = self._navigate_to_with_mapmanager(state, state.exploration_resume_position, reach_adjacent=False)
                if direction and self._is_direction_clear_in_obs(state, direction):
                    return self._actions.move.Move(direction)
                else:
                    # Can't reach resume position - clear it and continue
                    self._logger.warning(f"  GATHER: Can't reach resume position {state.exploration_resume_position}, clearing and continuing")
                    state.exploration_resume_position = None

        # PRIORITY 1: If no charger found yet, SEARCH FOR CHARGER FIRST
        if not state.found_initial_charger:
            self._logger.debug(f"  GATHER: Searching for INITIAL CHARGER (have {len(state.discovered_chargers)} discovered)")

            # Check if charger visible in observation
            result = self._find_station_direction_in_obs(state, "charger")
            if result is not None:
                charger_direction, _ = result
                if self._is_direction_clear_in_obs(state, charger_direction):
                    self._logger.info(f"  GATHER: Charger visible in direction {charger_direction}, moving toward it")
                    return self._actions.move.Move(charger_direction)

            # Check if charger adjacent
            adj_charger = self._find_station_adjacent_in_obs(state, "charger")
            if adj_charger is not None:
                self._logger.info(f"  GATHER: Charger adjacent in direction {adj_charger}, moving onto it")
                state.using_object_this_step = True
                return self._actions.move.Move(adj_charger)

            # Not visible - explore to find charger
            self._logger.debug(f"  GATHER: No charger visible, exploring to find one")
            return self._explore_observation_only(state)

        # PRIORITY 0.5: ENERGY SAFETY - if we're too far from charger, navigate back BEFORE gathering
        # This prevents getting trapped in dead-ends far from chargers
        if state.discovered_chargers:
            nearest_charger = min(
                state.discovered_chargers,
                key=lambda pos: abs(pos[0] - state.row) + abs(pos[1] - state.col)
            )
            dist_to_charger = abs(nearest_charger[0] - state.row) + abs(nearest_charger[1] - state.col)

            # Use EnergyManager for map-aware safe distance calculation
            max_safe_distance = self.energy.calculate_safe_radius(state)

            if dist_to_charger > max_safe_distance:
                self._logger.info(f"  GATHER: TOO FAR from charger ({dist_to_charger} > {max_safe_distance}, energy={state.energy}), navigating back for safety")

                # ESCAPE NARROW CORRIDORS: If stuck (many failed moves), backtrack aggressively
                # to escape narrow corridors and reach areas where lateral exploration is possible
                if state.consecutive_failed_moves >= 5:
                    self._logger.info(f"  GATHER: STUCK in corridor ({state.consecutive_failed_moves} fails), aggressive backtrack to charger")
                    # Go straight to charger to escape the corridor
                    dr = nearest_charger[0] - state.row
                    dc = nearest_charger[1] - state.col
                    direction = "north" if abs(dr) > abs(dc) and dr < 0 else \
                               "south" if abs(dr) > abs(dc) and dr > 0 else \
                               "east" if dc > 0 else "west"

                    if self._is_direction_clear_in_obs(state, direction):
                        return self._actions.move.Move(direction)

                    # Primary blocked - try alternatives
                    for alt_dir in ["north", "south", "east", "west"]:
                        if alt_dir != direction and self._is_direction_clear_in_obs(state, alt_dir):
                            return self._actions.move.Move(alt_dir)

                    return self._actions.noop.Noop()

                # LATERAL BACKTRACKING: Not stuck yet - try lateral exploration while moving closer
                dr = nearest_charger[0] - state.row
                dc = nearest_charger[1] - state.col

                # OSCILLATION FIX: Use committed direction for lateral movement
                # Only switch between lateral/direct when we've moved enough steps in current mode
                should_try_lateral = (state.committed_direction_steps < 3) or (state.step_count % 8 < 6)

                if should_try_lateral:
                    # Lateral exploration phase - try perpendicular directions
                    if abs(dr) > abs(dc):
                        # Charger is mostly vertical - explore horizontally
                        lateral_dirs = ["east", "west"]
                        primary_dir = "north" if dr < 0 else "south"
                    else:
                        # Charger is mostly horizontal - explore vertically
                        lateral_dirs = ["north", "south"]
                        primary_dir = "east" if dc > 0 else "west"

                    # Prefer continuing in committed direction if it's lateral
                    if state.committed_exploration_direction in lateral_dirs:
                        # Move committed direction to front
                        lateral_dirs = [state.committed_exploration_direction] + [d for d in lateral_dirs if d != state.committed_exploration_direction]

                    # Try lateral directions first (for area coverage)
                    for lat_dir in lateral_dirs:
                        if self._is_direction_clear_in_obs(state, lat_dir):
                            # Check if this lateral move keeps us within safe distance
                            deltas = {"north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1)}
                            ddr, ddc = deltas[lat_dir]
                            new_r, new_c = state.row + ddr, state.col + ddc
                            new_dist = abs(nearest_charger[0] - new_r) + abs(nearest_charger[1] - new_c)

                            if new_dist <= max_safe_distance + 2:  # Allow slight overshoot for exploration
                                self._logger.info(f"  GATHER: LATERAL backtrack {lat_dir} (exploring while returning)")
                                return self._actions.move.Move(lat_dir)

                    # Lateral directions didn't work - fall through to direct navigation
                    self._logger.debug(f"  GATHER: Lateral backtrack failed, using direct navigation")

                # Direct navigation toward charger (every 4th step or when lateral fails)
                direction = "north" if abs(dr) > abs(dc) and dr < 0 else \
                           "south" if abs(dr) > abs(dc) and dr > 0 else \
                           "east" if dc > 0 else "west"

                if self._is_direction_clear_in_obs(state, direction):
                    self._logger.info(f"  GATHER: Moving {direction} toward charger at {nearest_charger}")
                    return self._actions.move.Move(direction)

                # Try alternative directions sorted by distance reduction
                alt_directions = ["north", "south", "east", "west"]
                alt_directions.remove(direction)

                def dist_after_move(d):
                    deltas = {"north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1)}
                    ddr, ddc = deltas[d]
                    new_r, new_c = state.row + ddr, state.col + ddc
                    return abs(nearest_charger[0] - new_r) + abs(nearest_charger[1] - new_c)

                alt_directions.sort(key=dist_after_move)

                for alt_dir in alt_directions:
                    if self._is_direction_clear_in_obs(state, alt_dir):
                        self._logger.info(f"  GATHER: Primary blocked, trying {alt_dir}")
                        return self._actions.move.Move(alt_dir)

                # All blocked - use noop
                self._logger.warning(f"  GATHER: TOO FAR but all directions blocked! Using noop")
                return self._actions.noop.Noop()

        # PRIORITY 1: After finding charger, GO TO IT if energy is actually LOW
        # Only navigate to charger when below recharge_low, not when below recharge_high
        # recharge_high is for "stay on charger until full", not "start going to charger"
        recharge_low = self._get_recharge_low(state)
        recharge_high = self._get_recharge_high(state)
        if state.energy < recharge_low:
            self._logger.info(f"  GATHER: Energy LOW ({state.energy} < {recharge_low}), navigating to charger")

            # First check if we're ALREADY standing on a charger
            # If so, stay (noop) to continue charging
            standing_on_charger = False
            if state.current_obs and state.current_obs.tokens:
                center_pos = (self._obs_hr, self._obs_wr)
                for tok in state.current_obs.tokens:
                    if tok.location == center_pos and tok.feature.name == "tag":
                        tag_name = self._tag_names.get(tok.value, "").lower()
                        if "charger" in tag_name:
                            standing_on_charger = True
                            break

            if standing_on_charger:
                self._logger.info(f"  GATHER: Standing ON charger, staying (noop) to charge to {recharge_high}")
                state.using_object_this_step = True
                return self._actions.noop.Noop()

            # IMPORTANT: Check if charger ADJACENT (before checking visible)
            # Move onto adjacent charger (don't set using_object_this_step - let position update!)
            adj_charger = self._find_station_adjacent_in_obs(state, "charger")
            if adj_charger is not None:
                self._logger.info(f"  GATHER: Charger ADJACENT in direction {adj_charger}, moving onto it")
                return self._actions.move.Move(adj_charger)

            # Check if charger visible (but not adjacent) in observation
            result = self._find_station_direction_in_obs(state, "charger")
            if result is not None:
                charger_direction, _ = result
                if self._is_direction_clear_in_obs(state, charger_direction):
                    self._logger.debug(f"  GATHER: Charger visible (not adjacent) in direction {charger_direction}, moving toward it")
                    return self._actions.move.Move(charger_direction)

            # Charger not visible - navigate to nearest discovered charger using observation-only
            # Direction toward charger
            if state.discovered_chargers:
                nearest_charger = self._find_nearest_charger(state)
                dr = nearest_charger[0] - state.row
                dc = nearest_charger[1] - state.col

                # Pick direction that reduces distance most
                if abs(dr) > abs(dc):
                    # Vertical movement more important
                    direction = "north" if dr < 0 else "south"
                else:
                    # Horizontal movement more important
                    direction = "east" if dc > 0 else "west"

                # Check if that direction is clear
                if self._is_direction_clear_in_obs(state, direction):
                    self._logger.debug(f"  GATHER: Moving {direction} toward nearest charger at {nearest_charger}")
                    return self._actions.move.Move(direction)

                # Try perpendicular direction
                alt_directions = []
                if abs(dr) > abs(dc):
                    # Was trying vertical, try horizontal
                    alt_directions = ["east" if dc > 0 else "west", "west" if dc > 0 else "east"]
                else:
                    # Was trying horizontal, try vertical
                    alt_directions = ["north" if dr < 0 else "south", "south" if dr < 0 else "north"]

                for alt_dir in alt_directions:
                    if self._is_direction_clear_in_obs(state, alt_dir):
                        self._logger.debug(f"  GATHER: Primary direction blocked, trying {alt_dir}")
                        return self._actions.move.Move(alt_dir)

            # All directions blocked or no chargers - explore
            self._logger.debug(f"  GATHER: Can't navigate to charger, exploring")
            return self._explore_observation_only(state)

        # Energy is sufficient (>= recharge_low) - can now gather resources!
        self._logger.debug(f"  GATHER: Energy sufficient ({state.energy} >= {recharge_low}), gathering resources")

        # OPPORTUNISTIC CHARGING: If energy getting low and charger visible, use it
        # This prevents entering dedicated RECHARGE phase and oscillating to charger
        # Critical for energy_starved missions where exploration must continue
        if state.energy < 70:  # Charge opportunistically when below 70
            # Check if charger is adjacent and ready
            adj_charger = self._find_station_adjacent_in_obs(state, "charger")
            if adj_charger is not None:
                state.using_object_this_step = True
                return self._actions.move.Move(adj_charger)

            # If energy very low (< 30), actively navigate to nearest known charger
            if state.energy < 30:
                # First check if charger visible in observation
                result = self._find_station_direction_in_obs(state, "charger")
                if result is not None:
                    charger_direction, _ = result
                    if self._is_direction_clear_in_obs(state, charger_direction):
                        return self._actions.move.Move(charger_direction)

                # Not visible but we have discovered chargers - navigate to nearest
                if state.discovered_chargers:
                    nearest_charger = self._find_nearest_charger(state)
                    action = self._move_towards(state, nearest_charger, reach_adjacent=True)
                    if action.name != "noop":
                        return action

        # EXPLORATION MODE: If not all resource types discovered yet, focus on pure exploration
        # Don't try to collect resources until we know where all 4 types are located
        # BUT: On large maps, limit exploration time to avoid spending entire episode searching
        all_types_found = (
            len(self.map_manager.carbon_extractors) > 0 and
            len(self.map_manager.oxygen_extractors) > 0 and
            len(self.map_manager.germanium_extractors) > 0 and
            len(self.map_manager.silicon_extractors) > 0
        )

        # Determine exploration time budget based on map size
        is_large_map = state.mission_profile and state.mission_profile.map_size == "large"
        exploration_budget = 400 if is_large_map else 200

        if not all_types_found and state.step_count < exploration_budget:
            if state.step_count % 50 == 0:  # Log every 50 steps
                c_count = len(self.map_manager.carbon_extractors)
                o_count = len(self.map_manager.oxygen_extractors)
                g_count = len(self.map_manager.germanium_extractors)
                s_count = len(self.map_manager.silicon_extractors)
                self._logger.info(f"Step {state.step_count}: GATHER: EXPLORATION MODE - searching for all resource types (C={c_count} O={o_count} G={g_count} Si={s_count}), budget={exploration_budget}")
            return self._explore(state)
        elif not all_types_found:
            # Exploration budget exhausted - proceed with collection using available extractors
            c_count = len(self.map_manager.carbon_extractors)
            o_count = len(self.map_manager.oxygen_extractors)
            g_count = len(self.map_manager.germanium_extractors)
            s_count = len(self.map_manager.silicon_extractors)
            self._logger.warning(f"Step {state.step_count}: GATHER: Exploration budget exhausted ({exploration_budget} steps), proceeding with available extractors: C={c_count} O={o_count} G={g_count} Si={s_count}")
            # Fall through to collection mode

        # COLLECTION MODE: All resource types found, now collect from extractors
        # Find which resources we still need
        deficits = self._calculate_deficits(state)
        self._logger.debug(f"  GATHER: Resource deficits: {deficits}")
        if all(d <= 0 for d in deficits.values()):
            self._logger.debug(f"  GATHER: All resources satisfied, nooping")
            return self._actions.noop.Noop()

        # Sort resources by deficit - prioritize whichever we need most
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
        self._logger.debug(f"  GATHER: Priority order: {priority_order}")

        # Priority 1: Check if any READY extractor is visible and adjacent
        for resource_type in priority_order:
            if deficits.get(resource_type, 0) <= 0:
                continue  # Don't need this resource
            result = self._find_ready_extractor_in_obs(state, resource_type)
            if result is not None:
                obs_direction, _cooldown = result
                self._logger.info(f"  GATHER: Found READY {resource_type} extractor adjacent in direction {obs_direction}, using it!")
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
                    germanium_adjacent_on_cooldown = True
                    continue  # Try other resources first
                # For other resources on cooldown, move away to let them refresh
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

        # STUCK RECOVERY: If stuck, try to escape
        # Light stuck (5-9 failures): try observation-only exploration
        # Heavy stuck (10+ failures) ON LARGE MAPS: navigate back to charger to reset position
        if state.stuck_recovery_active or state.consecutive_failed_moves >= 5:
            # HEAVY STUCK: Navigate back to charger to reset
            # Use mission-aware stuck threshold
            stuck_threshold = state.mission_profile.stuck_threshold if state.mission_profile else 10
            if state.consecutive_failed_moves >= stuck_threshold and state.discovered_chargers:
                self._logger.info(f"  GATHER: HEAVILY STUCK ({state.consecutive_failed_moves} fails >= {stuck_threshold}), navigating to charger to reset")
                nearest_charger = self._find_nearest_charger(state)
                dr = nearest_charger[0] - state.row
                dc = nearest_charger[1] - state.col

                # Pick direction that reduces distance most
                if abs(dr) > abs(dc):
                    direction = "north" if dr < 0 else "south"
                else:
                    direction = "east" if dc > 0 else "west"

                if self._is_direction_clear_in_obs(state, direction):
                    self._logger.debug(f"  GATHER: Moving {direction} toward charger at {nearest_charger}")
                    return self._actions.move.Move(direction)

                # Try perpendicular
                alt_directions = []
                if abs(dr) > abs(dc):
                    alt_directions = ["east" if dc > 0 else "west", "west" if dc > 0 else "east"]
                else:
                    alt_directions = ["north" if dr < 0 else "south", "south" if dr < 0 else "north"]

                for alt_dir in alt_directions:
                    if self._is_direction_clear_in_obs(state, alt_dir):
                        self._logger.debug(f"  GATHER: Primary blocked, trying {alt_dir}")
                        return self._actions.move.Move(alt_dir)

            # LIGHT STUCK: Try observation-only exploration
            return self._explore_observation_only(state)

        # Priority 3: Navigate to known extractor OR explore if we don't have it
        # CRITICAL: Only navigate to extractors for resources we know about
        # If we need a resource but have NO extractors for it, EXPLORE to find it!
        top_priority_resource = None
        for res in priority_order:
            if deficits.get(res, 0) > 0:
                top_priority_resource = res
                break

        self._logger.debug(f"Step {state.step_count}: GATHER DECISION: top_priority={top_priority_resource}, pos=({state.row},{state.col})")

        if top_priority_resource:
            extractor = self._find_nearest_extractor(state, top_priority_resource)
            if extractor is not None:
                dist = abs(extractor.position[0] - state.row) + abs(extractor.position[1] - state.col)
                self._logger.info(f"Step {state.step_count}: GATHER: Navigating to {top_priority_resource} extractor at {extractor.position} (dist={dist})")

                # Try MapManager pathfinding first (uses complete map knowledge)
                direction = self._navigate_to_with_mapmanager(state, extractor.position, reach_adjacent=True)
                if direction:
                    if self._is_direction_clear_in_obs(state, direction):
                        self._logger.debug(f"Step {state.step_count}: GATHER: Moving {direction} toward extractor (MapManager)")
                        return self._actions.move.Move(direction)
                    else:
                        self._logger.warning(f"Step {state.step_count}: GATHER: MapManager suggested {direction} but BLOCKED in obs, trying fallback pathfinding")

                # Fall back to observation-based pathfinding
                self._logger.debug(f"Step {state.step_count}: GATHER: Falling back to observation-based pathfinding")
                action = self._move_towards(state, extractor.position, reach_adjacent=True)
                self._logger.debug(f"Step {state.step_count}: GATHER: Fallback pathfinding returned {action.name}")
                return action
            else:
                # We need this resource but have NO extractors for it - EXPLORE!
                self._logger.info(f"Step {state.step_count}: GATHER: Need {top_priority_resource} but no extractors found - EXPLORING")
                return self._explore(state)

        # Fallback: all resources satisfied (shouldn't reach here)
        self._logger.debug(f"  GATHER: All resources satisfied, nooping")
        return self._actions.noop.Noop()

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

        # STUCK RECOVERY: If stuck, use observation-only exploration
        # (Pathfinding uses occupancy map which may have errors)
        if state.stuck_recovery_active or state.consecutive_failed_moves >= 5:
            return self._explore_observation_only(state)

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
        """Recharge at the charger.

        CRITICAL: Must STAY on charger (noop) to charge, not keep moving!
        """
        recharge_high = self._get_recharge_high(state)
        self._logger.debug(f"  RECHARGE: energy={state.energy} (target={recharge_high})")

        # First check if we're CURRENTLY standing on a charger
        # Check center of observation (where we are) for charger tag
        standing_on_charger = False
        current_charger_pos = None
        if state.current_obs and state.current_obs.tokens:
            center_pos = (self._obs_hr, self._obs_wr)
            for tok in state.current_obs.tokens:
                if tok.location == center_pos and tok.feature.name == "tag":
                    tag_name = self._tag_names.get(tok.value, "").lower()
                    if "charger" in tag_name:
                        standing_on_charger = True
                        current_charger_pos = (state.row, state.col)
                        break

        # If standing on charger and not fully charged, STAY (noop) to charge
        if standing_on_charger and state.energy < recharge_high:
            self._logger.info(f"  RECHARGE: ON CHARGER at {current_charger_pos}, staying (noop) to charge (energy {state.energy} → {recharge_high})")
            state.using_object_this_step = True

            # CHARGER QUALITY TRACKING: Record successful use
            if current_charger_pos and current_charger_pos in state.charger_info:
                state.charger_info[current_charger_pos].times_successfully_used += 1
                self._logger.debug(f"  CHARGER: Successfully used charger at {current_charger_pos} "
                                 f"(success_rate={state.charger_info[current_charger_pos].success_rate:.2f})")

            return self._actions.noop.Noop()

        if standing_on_charger:
            self._logger.info(f"  RECHARGE: ON CHARGER but fully charged (energy={state.energy} >= {recharge_high})")

        # Not on charger or fully charged - find one
        # Check if charger adjacent - move onto it
        # CRITICAL: Don't set using_object_this_step - let position update so we can move ONTO charger!
        # BUT: If we've been stuck trying to move onto an adjacent charger for many steps, skip it and try navigation instead
        if state.consecutive_failed_moves < 5:  # Only try adjacent charger if not stuck
            adj_charger = self._find_station_adjacent_in_obs(state, "charger")
            if adj_charger is not None:
                self._logger.info(f"  RECHARGE: Charger adjacent in direction {adj_charger}, moving onto it")
                return self._actions.move.Move(adj_charger)
        else:
            self._logger.debug(f"  RECHARGE: Skipping adjacent charger check (stuck: {state.consecutive_failed_moves} failed moves)")

        # Check if charger visible but not adjacent - navigate to it
        # BUT: Skip if we're stuck (same issue as adjacent charger - might be inaccessible)
        if state.consecutive_failed_moves < 5:
            result = self._find_station_direction_in_obs(state, "charger")
            if result is not None:
                charger_direction, _ = result
                if self._is_direction_clear_in_obs(state, charger_direction):
                    self._logger.debug(f"  RECHARGE: Charger visible in direction {charger_direction}, navigating")
                    return self._actions.move.Move(charger_direction)
        else:
            self._logger.debug(f"  RECHARGE: Skipping visible charger check (stuck: {state.consecutive_failed_moves} failed moves)")

        # Charger not visible - navigate to discovered charger using intelligent selection
        # CHARGER QUALITY TRACKING: Prefer reliable chargers, avoid stuck patterns
        if state.discovered_chargers:
            # INTELLIGENT CHARGER SELECTION based on reliability and distance
            target_charger = self._select_best_charger(state)

            # Track that we're approaching this charger
            if target_charger != state.current_recharge_target:
                # New target - reset approach tracking
                state.current_recharge_target = target_charger
                if target_charger in state.charger_info:
                    state.charger_info[target_charger].times_approached += 1
                    state.charger_info[target_charger].last_attempt_step = state.step_count
                    self._logger.debug(f"  CHARGER: Approaching charger at {target_charger} "
                                     f"(attempts={state.charger_info[target_charger].times_approached}, "
                                     f"success_rate={state.charger_info[target_charger].success_rate:.2f})")

            dr = target_charger[0] - state.row
            dc = target_charger[1] - state.col

            self._logger.debug(f"  RECHARGE: Navigating to charger at {target_charger} (current pos=({state.row},{state.col}), dr={dr}, dc={dc})")

            # CRITICAL: Check for stuck status FIRST - escape oscillation and dead-ends
            # When stuck in RECHARGE, agent is trying to reach charger but trapped
            if state.consecutive_failed_moves >= 5:
                self._logger.warning(f"  RECHARGE: STUCK ({state.consecutive_failed_moves} fails) - attempting recovery")

                # CRITICAL FIX: After extended stuck (20+ fails), switch to EXPLORATION
                # This prevents infinite oscillation when all chargers are unreachable
                if state.consecutive_failed_moves >= 20:
                    self._logger.error(f"  RECHARGE: SEVERELY STUCK ({state.consecutive_failed_moves} fails) - "
                                     f"switching to EXPLORATION to find alternate path")
                    # Switch to exploration mode to try to navigate around obstacles
                    return self._explore(state)

                # Try perpendicular directions to find alternate path to charger
                if abs(dr) > abs(dc):
                    # Primarily moving vertically - try horizontal alternatives
                    alt_directions = ["east", "west", "south" if dr > 0 else "north"]
                else:
                    # Primarily moving horizontally - try vertical alternatives
                    alt_directions = ["north", "south", "west" if dc < 0 else "east"]

                import random
                random.shuffle(alt_directions)

                # OSCILLATION FIX: Check if direction leads to recently visited position
                dir_offsets = {"north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1)}
                recent_positions = set(state.position_history[-5:])  # Last 5 positions

                # Try alternative directions, avoiding recent positions
                for alt_dir in alt_directions:
                    if not self._is_direction_clear_in_obs(state, alt_dir):
                        continue

                    # Check if this would revisit a recent position (oscillation)
                    dr_alt, dc_alt = dir_offsets[alt_dir]
                    target_pos = (state.row + dr_alt, state.col + dc_alt)

                    if target_pos in recent_positions:
                        self._logger.debug(f"  STUCK RECOVERY: Skipping {alt_dir} - would revisit {target_pos}")
                        continue

                    self._logger.info(f"  STUCK RECOVERY: Trying alternative route {alt_dir}")
                    return self._actions.move.Move(alt_dir)

                # No non-oscillating alternatives found - try ANY clear direction (even if revisits)
                all_dirs = ["north", "south", "east", "west"]
                random.shuffle(all_dirs)
                for desperation_dir in all_dirs:
                    if self._is_direction_clear_in_obs(state, desperation_dir):
                        self._logger.warning(f"  STUCK RECOVERY: DESPERATION - trying {desperation_dir} (may oscillate)")
                        return self._actions.move.Move(desperation_dir)

                # Completely stuck - use exploration to escape
                self._logger.error(f"  STUCK RECOVERY: ALL DIRECTIONS BLOCKED - using exploration")
                return self._explore(state)

            # Use MapManager pathfinding for intelligent navigation to charger
            direction = self._navigate_to_with_mapmanager(state, target_charger, reach_adjacent=False)

            if direction:
                self._logger.debug(f"  RECHARGE: Moving {direction} toward charger (MapManager pathfinding)")
                return self._actions.move.Move(direction)

            # Pathfinding failed - fall back to greedy navigation
            self._logger.debug(f"  RECHARGE: No path to charger found, trying greedy navigation")

            # Pick direction that reduces distance most
            if abs(dr) > abs(dc):
                # Vertical movement more important
                direction = "north" if dr < 0 else "south"
            else:
                # Horizontal movement more important
                direction = "east" if dc > 0 else "west"

            # Check if that direction is clear
            if self._is_direction_clear_in_obs(state, direction):
                self._logger.debug(f"  RECHARGE: Moving {direction} toward charger (greedy fallback)")
                return self._actions.move.Move(direction)

            # Try alternative directions
            alt_directions = []
            if abs(dr) > abs(dc):
                alt_directions = ["east" if dc > 0 else "west", "west" if dc > 0 else "east"]
            else:
                alt_directions = ["north" if dr < 0 else "south", "south" if dr < 0 else "north"]

            for alt_dir in alt_directions:
                if self._is_direction_clear_in_obs(state, alt_dir):
                    self._logger.debug(f"  RECHARGE: Primary blocked, trying {alt_dir}")
                    return self._actions.move.Move(alt_dir)

        # All directions blocked or no chargers - explore to find one
        self._logger.debug(f"  RECHARGE: Can't navigate to charger, exploring (have {len(state.discovered_chargers)} discovered)")
        return self._explore_observation_only(state)

    def _find_nearest_charger(self, state: HarvestState) -> Optional[tuple[int, int]]:
        """Find the nearest charger using MapManager.

        Returns:
            Position of nearest charger, or None if none found.
        """
        if self.map_manager is None:
            # Fallback if MapManager not initialized yet
            if state.discovered_chargers:
                return min(
                    state.discovered_chargers,
                    key=lambda pos: abs(pos[0] - state.row) + abs(pos[1] - state.col)
                )
            return state.stations.get("charger")

        # Use MapManager for complete map knowledge
        return self.map_manager.get_nearest_object(
            current_pos=(state.row, state.col),
            object_type="charger"
        )

    def _find_nearest_assembler(self, state: HarvestState) -> Optional[tuple[int, int]]:
        """Find the nearest assembler using MapManager.

        Returns:
            Position of nearest assembler, or None if none found.
        """
        if self.map_manager is None:
            return state.stations.get("assembler")

        return self.map_manager.get_nearest_object(
            current_pos=(state.row, state.col),
            object_type="assembler"
        )

    def _find_nearest_chest(self, state: HarvestState) -> Optional[tuple[int, int]]:
        """Find the nearest chest using MapManager.

        Returns:
            Position of nearest chest, or None if none found.
        """
        if self.map_manager is None:
            return state.stations.get("chest")

        return self.map_manager.get_nearest_object(
            current_pos=(state.row, state.col),
            object_type="chest"
        )

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
        """Find the nearest available extractor of a given type using MapManager.

        Args:
            resource_type: Type of resource ("carbon", "oxygen", "germanium", "silicon")

        Returns:
            ExtractorInfo if found, None otherwise.
        """
        # Try MapManager first (complete map knowledge)
        if self.map_manager is not None:
            extractor_pos = self.resources.find_nearest_available_extractor(
                state, resource_type, self.map_manager
            )
            if extractor_pos:
                # Create ExtractorInfo from position
                # Note: MapManager doesn't track detailed extractor state
                return ExtractorInfo(
                    position=extractor_pos,
                    resource_type=resource_type,
                    last_seen_step=state.step_count,
                    times_used=0,
                    cooldown_remaining=0,
                    clipped=False
                )

        # Fallback to old ExtractorInfo system
        extractors = state.extractors.get(resource_type, [])
        available = [
            e for e in extractors
            if not e.clipped
            and e.remaining_uses > 0
            and e.position not in state.used_extractors
        ]

        if not available:
            return None

        def distance(ext: ExtractorInfo) -> int:
            return abs(ext.position[0] - state.row) + abs(ext.position[1] - state.col)

        return min(available, key=distance)

    def _navigate_to_with_mapmanager(
        self,
        state: HarvestState,
        target: tuple[int, int],
        reach_adjacent: bool = False
    ) -> Optional[str]:
        """Navigate to target using MapManager's complete map knowledge.

        Uses BFS pathfinding with MapManager.is_traversable() which knows the
        entire explored map, not just current observation.

        Args:
            state: Current agent state
            target: Target position (row, col)
            reach_adjacent: Whether to stop adjacent to target instead of on it

        Returns:
            Direction to move ("north", "south", "east", "west"), or None if no path.
        """
        if self.map_manager is None:
            return None

        # DEBUG: Check target cell type
        from .map import MapCellType
        target_cell = self.map_manager.grid[target[0]][target[1]]
        self._logger.debug(f"  PATHFIND: Target {target} is {target_cell.name}, traversable={self.map_manager.is_traversable(target[0], target[1])}")

        # Compute goal cells
        if reach_adjacent:
            goals = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = target[0] + dr, target[1] + dc
                if self.map_manager.is_traversable(nr, nc):
                    goals.append((nr, nc))
            if not goals:
                goals = [target]  # Fallback if no adjacent cells traversable
                self._logger.warning(f"  PATHFIND: No adjacent cells traversable around {target}, using target itself")
        else:
            goals = [target]

        self._logger.debug(f"  PATHFIND: Goals={goals}, start={state.row, state.col}")

        # DEBUG: Check cells around agent
        agent_vicinity = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = state.row + dr, state.col + dc
            if 0 <= nr < state.map_height and 0 <= nc < state.map_width:
                cell = self.map_manager.grid[nr][nc]
                trav = self.map_manager.is_traversable(nr, nc)
                agent_vicinity.append(f"{cell.name[0]}:{trav}")
        self._logger.debug(f"  PATHFIND: Agent vicinity NSEW: {agent_vicinity}")

        # Use pathfinding with MapManager's traversability
        path = shortest_path(
            state=state,
            start=(state.row, state.col),
            goals=goals,
            allow_goal_block=False,
            cell_type=CellType,
            is_traversable_fn=lambda r, c: self.map_manager.is_traversable(r, c)
        )

        if not path or len(path) < 1:
            self._logger.warning(f"  PATHFIND: No path found from {state.row, state.col} to {target}")
            return None

        self._logger.debug(f"  PATHFIND: Path found, length={len(path)}, next={path[0]}")

        # Convert next position to direction
        next_pos = path[0]  # First step in path
        dr = next_pos[0] - state.row
        dc = next_pos[1] - state.col

        if dr == -1:
            return "north"
        elif dr == 1:
            return "south"
        elif dc == -1:
            return "west"
        elif dc == 1:
            return "east"

        return None  # Invalid move

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
            self._logger.warning(f"Step {state.step_count}: PATHFIND FAILED to {target}, trying any clear direction (THIS CAUSES OSCILLATION!)")
            for direction in ["north", "south", "east", "west"]:
                if self._is_direction_clear_in_obs(state, direction):
                    self._logger.debug(f"Step {state.step_count}: PATHFIND FAILED: Moving {direction} (random)")
                    return self._actions.move.Move(direction)
            self._logger.error(f"Step {state.step_count}: PATHFIND FAILED: All directions blocked, nooping")
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

    def _explore_observation_only(self, state: HarvestState) -> Action:
        """Explore using ONLY observation-based movement (no pathfinding).

        ENERGY-AWARE EXPLORATION: Only explores within safe radius of chargers.
        Uses breadth-first strategy to avoid dead-end traps.
        """
        # FORCED EXPLORATION: When badly stuck, force movement in specific direction
        if state.forced_exploration_direction:
            forced_dir = state.forced_exploration_direction
            if self._is_direction_clear_in_obs(state, forced_dir):
                self._logger.info(f"  EXPLORE: Using forced direction {forced_dir} to escape stuck state")
                return self._actions.move.Move(forced_dir)
            else:
                # Forced direction blocked, clear it and proceed with normal exploration
                self._logger.info(f"  EXPLORE: Forced direction {forced_dir} blocked, trying alternatives")
                state.forced_exploration_direction = None

        # FRONTIER EXPLORATION: Try using MapManager to target unexplored boundary
        if self.map_manager is not None and len(state.discovered_chargers) > 0:
            # Find nearest unexplored frontier cell
            frontier = self.exploration.find_nearest_frontier_cell(state, self.map_manager)

            if frontier:
                self._logger.info(f"  EXPLORE: Targeting frontier cell at {frontier}")

                # Try pathfinding to frontier using MapManager
                direction = self._navigate_to_with_mapmanager(state, frontier, reach_adjacent=False)

                if direction:
                    if self._is_direction_clear_in_obs(state, direction):
                        self._logger.info(f"  EXPLORE: Moving {direction} toward frontier")
                        return self._actions.move.Move(direction)
                    else:
                        self._logger.debug(f"  EXPLORE: Frontier path blocked in obs, falling back to scoring")
                else:
                    self._logger.debug(f"  EXPLORE: No path to frontier found, falling back to scoring")

        dir_offsets = {"north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1)}

        # ENERGY SAFETY: Calculate safe exploration radius using EnergyManager
        # Automatically scales with map size and number of chargers
        nearest_charger_dist = float('inf')

        if state.discovered_chargers:
            nearest_charger = min(
                state.discovered_chargers,
                key=lambda pos: abs(pos[0] - state.row) + abs(pos[1] - state.col)
            )
            nearest_charger_dist = abs(nearest_charger[0] - state.row) + abs(nearest_charger[1] - state.col)

        # Use EnergyManager for map-aware safe distance calculation
        max_safe_distance_from_charger = self.energy.calculate_safe_radius(state)
        self._logger.debug(f"  EXPLORE: Safe radius={max_safe_distance_from_charger} (energy={state.energy}, chargers={len(state.discovered_chargers)})")

        # Check which directions are clear AND score them by exploration value
        direction_scores = []
        for direction in ["north", "south", "east", "west"]:
            if not self._is_direction_clear_in_obs(state, direction):
                continue  # Skip blocked directions

            # Calculate target position
            dr, dc = dir_offsets[direction]
            target_r, target_c = state.row + dr, state.col + dc

            # DEAD-END AVOIDANCE: Never re-enter marked dead-ends
            if (target_r, target_c) in state.dead_end_positions:
                self._logger.debug(f"  EXPLORE: Direction {direction} leads to dead-end at {(target_r, target_c)} - skipping")
                continue  # Skip this direction entirely

            # ENERGY SAFETY CHECK: Don't explore too far from charger
            if state.discovered_chargers:
                # Calculate distance from nearest charger to target position
                dist_to_charger = abs(nearest_charger[0] - target_r) + abs(nearest_charger[1] - target_c)

                # If moving this direction would take us too far from charger, PENALIZE it heavily
                if dist_to_charger > max_safe_distance_from_charger:
                    # Too far from charger - very low priority (but not zero, in case we're already far)
                    score = -100 + dist_to_charger  # Prefer directions that move us BACK toward charger
                    self._logger.debug(f"  EXPLORE: Direction {direction} too far from charger ({dist_to_charger} > {max_safe_distance_from_charger}), penalizing")
                else:
                    # Within safe radius - score normally
                    if (target_r, target_c) in state.explored_cells:
                        score = 0  # Already explored - low priority
                    else:
                        score = 10  # Unexplored - high priority

                    # OSCILLATION FIX: Momentum bonus for continuing in same direction
                    # This prevents flip-flopping between directions with similar scores
                    if direction == state.committed_exploration_direction and state.committed_direction_steps > 0:
                        momentum_bonus = min(15, state.committed_direction_steps * 3)  # Up to +15 for continuing
                        score += momentum_bonus
                        self._logger.debug(f"  EXPLORE: Momentum bonus +{momentum_bonus} for continuing {direction} (steps={state.committed_direction_steps})")

                    # SPIRAL EXPLORATION: Alternate between horizontal and vertical movement
                    # This prevents getting stuck in linear corridors
                    # CRITICAL: If we're near charger/start position, STRONGLY prefer east/west to escape corridors
                    near_charger = dist_to_charger < 10
                    if near_charger:
                        # Near charger - explore laterally to discover new areas
                        if direction in ["east", "west"]:
                            score += 20  # Very strong boost for lateral exploration
                            self._logger.info(f"  EXPLORE: Near charger, strongly boosting lateral {direction}")
                    elif state.step_count % 20 < 10:
                        # Horizontal sweep phase - prefer east/west
                        if direction in ["east", "west"]:
                            score += 5  # Boost horizontal directions
                            self._logger.debug(f"  EXPLORE: Horizontal sweep phase, boosting {direction}")
                    else:
                        # Vertical sweep phase - prefer north/south
                        if direction in ["north", "south"]:
                            score += 5  # Boost vertical directions
                            self._logger.debug(f"  EXPLORE: Vertical sweep phase, boosting {direction}")

                    # BREADTH-FIRST: Slightly prefer directions that keep us closer to charger
                    # This encourages expanding circles rather than deep exploration
                    score -= dist_to_charger * 0.1
            else:
                # No charger discovered yet - explore freely
                if (target_r, target_c) in state.explored_cells:
                    score = 0  # Already explored - low priority
                else:
                    score = 10  # Unexplored - high priority

                # Momentum bonus even without charger
                if direction == state.committed_exploration_direction and state.committed_direction_steps > 0:
                    momentum_bonus = min(15, state.committed_direction_steps * 3)
                    score += momentum_bonus

            # OSCILLATION FIX: Deterministic tie-breaking based on position
            # Avoid random component that causes instability
            # Use position-based offset so different positions prefer different directions
            position_offset = (state.row * 7 + state.col * 11) % 4
            if ["north", "south", "east", "west"].index(direction) == position_offset:
                score += 0.5  # Small deterministic bonus

            direction_scores.append((direction, score))

        if not direction_scores:
            # Completely surrounded in observation - all directions appear blocked
            # DESPERATION MODE: When stuck for 10+ steps, try moves anyway (observation might be wrong)
            if state.consecutive_failed_moves >= 10:
                # Rotate through directions to systematically try escaping
                all_dirs = ["north", "south", "east", "west"]
                desperation_idx = state.step_count % 4
                desperation_dir = all_dirs[desperation_idx]
                self._logger.warning(f"  EXPLORE: DESPERATION MODE - all directions blocked, trying {desperation_dir} anyway (stuck {state.consecutive_failed_moves} steps)")
                return self._actions.move.Move(desperation_dir)
            else:
                # Not stuck long enough - try noop (might be on charger)
                return self._actions.noop.Noop()

        # Pick highest-scoring direction (prefer unexplored)
        best_direction = max(direction_scores, key=lambda x: x[1])[0]

        # OSCILLATION FIX: Track committed direction for momentum
        if best_direction == state.committed_exploration_direction:
            # Continuing in same direction - increment counter
            state.committed_direction_steps += 1
        else:
            # Changed direction - reset counter
            state.committed_exploration_direction = best_direction
            state.committed_direction_steps = 1
            self._logger.info(f"  EXPLORE: Changed direction to {best_direction}")

        return self._actions.move.Move(best_direction)

    def _explore(self, state: HarvestState) -> Action:
        """Explore using frontier-based navigation + observation fallback.

        Strategy:
        1. Check for visible extractors we need (reactive)
        2. Use MapManager frontier-based pathfinding to reach unexplored boundaries
        3. Fall back to directional patrol only if pathfinding fails
        """
        # Priority 1: Check if we can see a READY extractor for a resource we NEED
        deficits = self._calculate_deficits(state)
        for resource_type in ["germanium", "carbon", "oxygen", "silicon"]:
            if deficits.get(resource_type, 0) <= 0:
                continue
            result = self._find_ready_extractor_in_obs(state, resource_type)
            if result is not None:
                obs_direction, _cooldown = result
                state.using_object_this_step = True
                return self._actions.move.Move(obs_direction)

        # Priority 2: Check if we can see ANY extractor we need (visible but not adjacent)
        for resource_type in ["germanium", "carbon", "oxygen", "silicon"]:
            if deficits.get(resource_type, 0) <= 0:
                continue
            nav_direction = self._find_extractor_direction_in_obs(state, resource_type)
            if nav_direction is not None:
                return self._actions.move.Move(nav_direction)

        # Priority 3: USE MAPMANAGER FRONTIER EXPLORATION (more efficient on large maps)
        # This uses complete map knowledge instead of just state.explored_cells
        if self.map_manager is not None and len(state.discovered_chargers) > 0:
            # OSCILLATION FIX: If we have a committed frontier target, stick with it for multiple steps
            # Only re-evaluate when we get close, stuck, or commitment expires
            committed_target = state.committed_frontier_target
            if committed_target and state.frontier_target_commitment_steps > 0:
                # Check if we're close to the target (within 3 cells)
                dist_to_target = abs(committed_target[0] - state.row) + abs(committed_target[1] - state.col)
                if dist_to_target <= 3:
                    # Close enough - clear commitment and find new target
                    self._logger.info(f"Step {state.step_count}: EXPLORE: Reached committed frontier {committed_target}, clearing commitment")
                    state.committed_frontier_target = None
                    state.frontier_target_commitment_steps = 0
                # CRITICAL: Clear commitment if stuck to avoid oscillating on unreachable frontiers
                elif state.consecutive_failed_moves >= 5:
                    self._logger.warning(f"Step {state.step_count}: EXPLORE: STUCK ({state.consecutive_failed_moves} fails) while pursuing frontier {committed_target}, clearing commitment to find alternate target")
                    state.committed_frontier_target = None
                    state.frontier_target_commitment_steps = 0
                else:
                    # Still far away and not stuck - continue toward committed target
                    direction = self._navigate_to_with_mapmanager(state, committed_target, reach_adjacent=False)
                    if direction and self._is_direction_clear_in_obs(state, direction):
                        state.frontier_target_commitment_steps -= 1
                        self._logger.debug(f"Step {state.step_count}: EXPLORE: Continuing toward committed frontier {committed_target} ({dist_to_target} away, {state.frontier_target_commitment_steps} steps left)")
                        return self._actions.move.Move(direction)
                    else:
                        # Path blocked - clear commitment and find new target
                        self._logger.info(f"Step {state.step_count}: EXPLORE: Path to committed frontier {committed_target} blocked, clearing commitment")
                        state.committed_frontier_target = None
                        state.frontier_target_commitment_steps = 0

            # Need to find new frontier target (no commitment or commitment cleared)
            # CRITICAL FIX: Find ALL frontier candidates and try pathfinding to each
            # until we find one that's REACHABLE (not just closest by distance)
            frontier_candidates = self._find_all_frontier_cells(state, self.map_manager, max_candidates=10)

            if frontier_candidates:
                self._logger.debug(f"Step {state.step_count}: EXPLORE: Found {len(frontier_candidates)} frontier candidates")

                # Try pathfinding to each frontier candidate until one succeeds
                for i, frontier in enumerate(frontier_candidates):
                    dist = abs(frontier[0] - state.row) + abs(frontier[1] - state.col)
                    self._logger.debug(f"Step {state.step_count}: EXPLORE: Trying frontier {i+1}/{len(frontier_candidates)} at {frontier} (dist={dist})")

                    # Try pathfinding to frontier using MapManager
                    direction = self._navigate_to_with_mapmanager(state, frontier, reach_adjacent=False)

                    if direction:
                        if self._is_direction_clear_in_obs(state, direction):
                            # OSCILLATION FIX: Commit to this frontier for multiple steps
                            # Commitment duration based on distance and map size
                            # Large maps with complex mazes need longer commitment to reach distant frontiers
                            map_size = state.mission_profile.map_size if state.mission_profile else "medium"
                            if map_size == "large":
                                # Large maps: dist / 2, capped at 30 (e.g., 60 cells away = 30 step commitment)
                                commitment_duration = min(30, max(5, dist // 2))
                            elif map_size == "medium":
                                # Medium maps: dist / 2.5, capped at 20 (e.g., 50 cells away = 20 steps)
                                commitment_duration = min(20, max(5, dist * 2 // 5))
                            else:
                                # Small maps: dist / 3, capped at 15 (quick re-evaluation)
                                commitment_duration = min(15, max(5, dist // 3))
                            state.committed_frontier_target = frontier
                            state.frontier_target_commitment_steps = commitment_duration
                            self._logger.info(f"Step {state.step_count}: EXPLORE: Found REACHABLE frontier at {frontier} (dist={dist}), committing for {commitment_duration} steps (map_size={map_size})")
                            return self._actions.move.Move(direction)
                        else:
                            self._logger.debug(f"Step {state.step_count}: EXPLORE: Frontier {frontier} path {direction} blocked in obs, trying next")
                    else:
                        self._logger.debug(f"Step {state.step_count}: EXPLORE: No path to frontier {frontier}, trying next")

                # All frontiers unreachable - try advanced maze navigation algorithms
                self._logger.warning(f"Step {state.step_count}: EXPLORE: All {len(frontier_candidates)} frontiers UNREACHABLE, trying advanced navigation")

                # Strategy 2: Wavefront expansion from nearest charger
                if state.discovered_chargers:
                    nearest_charger = self._find_nearest_charger(state)
                    wavefront_target = self.maze_navigator.get_systematic_exploration_target(
                        state, self.map_manager, nearest_charger
                    )
                    if wavefront_target:
                        direction = self._navigate_to_with_mapmanager(state, wavefront_target, reach_adjacent=False)
                        if direction and self._is_direction_clear_in_obs(state, direction):
                            self._logger.info(f"Step {state.step_count}: EXPLORE: Using wavefront expansion to {wavefront_target}")
                            return self._actions.move.Move(direction)

                # Strategy 3: Target largest unexplored region
                region_target = self.maze_navigator.find_largest_unexplored_region(state, self.map_manager)
                if region_target:
                    direction = self._navigate_to_with_mapmanager(state, region_target, reach_adjacent=False)
                    if direction and self._is_direction_clear_in_obs(state, direction):
                        self._logger.info(f"Step {state.step_count}: EXPLORE: Targeting largest unexplored region at {region_target}")
                        return self._actions.move.Move(direction)

                # Strategy 4: Wall-following (guaranteed to explore connected mazes)
                wall_follow_dir = self.maze_navigator.wall_follow_next_direction(state, self.map_manager)
                if wall_follow_dir and self._is_direction_clear_in_obs(state, wall_follow_dir):
                    self._logger.info(f"Step {state.step_count}: EXPLORE: Using wall-following, moving {wall_follow_dir}")
                    return self._actions.move.Move(wall_follow_dir)

                # All advanced strategies failed - fall through to quadrant navigation
                self._logger.warning(f"Step {state.step_count}: EXPLORE: All advanced strategies failed, falling back to quadrant navigation")
            else:
                self._logger.warning(f"Step {state.step_count}: EXPLORE: No frontier cells found in MapManager!")

        # Priority 4: Navigate to current exploration quadrant target
        # CRITICAL for quadrant_buildings missions
        quadrant_target = self._get_quadrant_target(state)
        if quadrant_target not in state.explored_cells:
            action = self._move_towards(state, quadrant_target, reach_adjacent=False)
            if action.name != "noop":
                return action

        # Priority 5: OLD frontier-based exploration (fallback for when MapManager not available)
        frontier_targets = self._find_frontier_targets(state)
        if frontier_targets:
            # CRITICAL: On large maps, pick FARTHEST frontier in current quadrant
            # This ensures we systematically explore the current quadrant
            is_large_map = state.observed_map_extent > 50
            if is_large_map:
                # Filter frontiers to current quadrant, or use all if none in quadrant
                quadrant_frontiers = [
                    t for t in frontier_targets
                    if self._is_in_current_quadrant(state, t)
                ]
                if quadrant_frontiers:
                    frontier_targets = quadrant_frontiers

                # Farthest frontier - maximizes exploration coverage
                target = max(frontier_targets, key=lambda t: abs(t[0] - state.row) + abs(t[1] - state.col))
            else:
                # Small maps: nearest is fine
                target = min(frontier_targets, key=lambda t: abs(t[0] - state.row) + abs(t[1] - state.col))

            action = self._move_towards(state, target, reach_adjacent=False)
            if action.name != "noop":
                return action

        # Fallback: No extractors visible and no good frontier targets
        # Use directional patrol pattern (existing behavior)
        patrol_patterns = [
            ["south", "east", "north", "west"],
            ["south", "west", "north", "east"],
            ["north", "west", "south", "east"],
            ["north", "east", "south", "west"],
        ]
        patrol_period = 50  # Optimal patrol period
        pattern_idx = (state.step_count // patrol_period) % 4
        preferred_order = patrol_patterns[pattern_idx]

        # Check which directions are clear using observation
        clear_directions = []
        for direction in ["north", "south", "east", "west"]:
            if self._is_direction_clear_in_obs(state, direction):
                clear_directions.append(direction)

        if not clear_directions:
            return self._actions.noop.Noop()

        # Pick first preferred direction that's clear
        for direction in preferred_order:
            if direction in clear_directions:
                return self._actions.move.Move(direction)

        return self._actions.move.Move(clear_directions[0])

    def _is_direction_clear_in_obs(self, state: HarvestState, direction: str) -> bool:
        """Check if a direction is clear based on observation.

        Returns True if:
        - No wall at target position
        - No other agent at target position
        - Extractors, stations ARE traversable (we move onto them to use them)
        - Empty space is traversable
        """
        # Safety: if no observation, return True (optimistic - let move verification handle it)
        # Being too conservative here causes all moves to fail
        if state.current_obs is None:
            return True

        dir_offsets = {"north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1)}
        dr, dc = dir_offsets[direction]
        target_obs_pos = (self._obs_hr + dr, self._obs_wr + dc)

        # Check all tokens at the target position
        for tok in state.current_obs.tokens:
            if tok.location == target_obs_pos and tok.feature.name == "tag":
                tag_name = self._tag_names.get(tok.value, "").lower()
                # Block ONLY on true obstacles: walls and agents
                if "wall" in tag_name or tag_name == "agent":
                    return False

        # No blocking object found - path is clear
        return True

    def _find_all_frontier_cells(
        self,
        state: HarvestState,
        map_manager: 'MapManager',
        max_candidates: int = 10
    ) -> list[tuple[int, int]]:
        """Find multiple frontier candidates sorted by distance.

        Returns up to max_candidates frontier cells, prioritized by Manhattan distance.
        This allows trying multiple frontiers when the closest one is unreachable.

        Args:
            state: Current agent state
            map_manager: MapManager with complete map grid
            max_candidates: Maximum number of candidates to return

        Returns:
            List of frontier positions, sorted by distance (closest first)
        """
        from .map import MapCellType

        frontier_candidates = []

        # Scan map for frontier cells
        # Adaptive search radius based on map size - larger maps need wider search
        map_dimension = max(state.map_height, state.map_width)
        if map_dimension > 200:
            search_radius = 150  # Large maps (500x500): search 300x300 window
        elif map_dimension > 100:
            search_radius = 75   # Medium maps: search 150x150 window
        else:
            search_radius = 50   # Small maps: search 100x100 window

        start_r = max(0, state.row - search_radius)
        end_r = min(state.map_height, state.row + search_radius + 1)
        start_c = max(0, state.col - search_radius)
        end_c = min(state.map_width, state.col + search_radius + 1)

        for r in range(start_r, end_r):
            for c in range(start_c, end_c):
                # Must be EXPLORED and TRAVERSABLE
                cell_type = map_manager.grid[r][c]
                if cell_type in (MapCellType.UNKNOWN, MapCellType.WALL, MapCellType.DEAD_END):
                    continue

                # Check if adjacent to any UNKNOWN cell
                is_frontier = False
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < state.map_height and 0 <= nc < state.map_width:
                        if map_manager.grid[nr][nc] == MapCellType.UNKNOWN:
                            is_frontier = True
                            break

                if is_frontier:
                    dist = abs(r - state.row) + abs(c - state.col)
                    frontier_candidates.append((dist, r, c))

        if not frontier_candidates:
            return []

        # Sort by distance and return top N
        frontier_candidates.sort(key=lambda x: x[0])
        return [(r, c) for (_, r, c) in frontier_candidates[:max_candidates]]

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

    def _get_quadrant_target(self, state: HarvestState) -> tuple[int, int]:
        """Get target position in the current exploration quadrant.

        CRITICAL for quadrant_buildings missions - ensures systematic coverage.
        """
        center = state.map_height // 2
        offset = state.map_height // 4  # Quarter of map size

        # Quadrant 0: NorthEast, 1: SouthEast, 2: SouthWest, 3: NorthWest
        if state.exploration_quadrant == 0:  # NE
            return (center - offset, center + offset)
        elif state.exploration_quadrant == 1:  # SE
            return (center + offset, center + offset)
        elif state.exploration_quadrant == 2:  # SW
            return (center + offset, center - offset)
        else:  # NW
            return (center - offset, center - offset)

    def _is_in_current_quadrant(self, state: HarvestState, pos: tuple[int, int]) -> bool:
        """Check if position is in the current exploration quadrant."""
        r, c = pos
        center = state.map_height // 2

        # Quadrant 0: NE (north=top=negative, east=right=positive)
        # Quadrant 1: SE, 2: SW, 3: NW
        if state.exploration_quadrant == 0:  # NE
            return r < center and c >= center
        elif state.exploration_quadrant == 1:  # SE
            return r >= center and c >= center
        elif state.exploration_quadrant == 2:  # SW
            return r >= center and c < center
        else:  # NW
            return r < center and c < center


class HarvestPolicy(MultiAgentPolicy):
    """Multi-agent policy wrapper for the harvest mission."""

    short_names = ["harvest_submission"]

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
