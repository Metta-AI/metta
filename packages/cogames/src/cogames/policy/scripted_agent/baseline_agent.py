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
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable, Optional, Tuple, Union

from cogames.policy import StatefulPolicyImpl
from mettagrid.config.mettagrid_config import CardinalDirection, CardinalDirections
from mettagrid.config.vibes import VIBE_BY_NAME
from mettagrid.policy.policy import MultiAgentPolicy, StatefulAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action
from mettagrid.simulator.interface import AgentObservation

if TYPE_CHECKING:
    pass

# Debug flag - set to True to enable detailed logging
DEBUG = False

# Sentinel for agent-centric features
AGENT_SENTINEL = 0x55


@dataclass
class BaselineHyperparameters:
    """Hyperparameters controlling baseline agent behavior."""

    # Energy management (recharge timing)
    recharge_threshold_low: int = 35  # Enter RECHARGE phase when energy < this
    recharge_threshold_high: int = 85  # Exit RECHARGE phase when energy >= this

    # Exploration strategy
    exploration_visit_weight: float = 15.0  # Multiplier for visit_score (higher = prefer unexplored)
    exploration_distance_penalty: float = 0.7  # Multiplier for distance (lower = willing to travel far)
    exploration_quadrant_bonus: float = 40.0  # Bonus for exploring opposite quadrant
    exploration_target_persistence: int = 12  # Steps to commit to exploration target

    # Stuck detection and escape
    stuck_detection_enabled: bool = True  # Enable loop detection
    stuck_escape_distance: int = 12  # Minimum distance for escape target


# Hyperparameter Presets for Ensemble Creation
BASELINE_HYPERPARAMETER_PRESETS = {
    "default": BaselineHyperparameters(
        recharge_threshold_low=35,  # Moderate energy management
        recharge_threshold_high=85,
        exploration_visit_weight=15.0,  # Moderate exploration preference
        exploration_distance_penalty=0.7,  # Moderate willingness to travel
        exploration_quadrant_bonus=40.0,  # Moderate cross-map exploration
        exploration_target_persistence=12,  # Moderate target persistence
        stuck_detection_enabled=True,
        stuck_escape_distance=12,
    ),
    "conservative": BaselineHyperparameters(
        recharge_threshold_low=50,  # Recharge early
        recharge_threshold_high=95,  # Stay charged
        exploration_visit_weight=10.0,  # Less willing to travel far for exploration
        exploration_distance_penalty=1.0,  # Higher penalty = prefer nearby
        exploration_quadrant_bonus=30.0,  # Lower cross-map bonus
        exploration_target_persistence=8,  # Switch targets more frequently
        stuck_detection_enabled=True,
        stuck_escape_distance=8,  # Shorter escape distance
    ),
    "aggressive": BaselineHyperparameters(
        recharge_threshold_low=20,  # Low energy tolerance
        recharge_threshold_high=80,  # Don't wait for full charge
        exploration_visit_weight=30.0,  # Strongly prefer unexplored areas
        exploration_distance_penalty=0.3,  # Very willing to travel far
        exploration_quadrant_bonus=70.0,  # Strong cross-map exploration
        exploration_target_persistence=15,  # Commit longer to distant targets
        stuck_detection_enabled=True,
        stuck_escape_distance=15,  # Longer escape distance
    ),
}


class CellType(Enum):
    """Occupancy map cell states."""

    FREE = 1  # Passable (can walk through)
    OBSTACLE = 2  # Impassable (walls, stations, extractors)


class Phase(Enum):
    """Goal-driven phases for the baseline agent."""

    GATHER = "gather"  # Collect resources (explore if needed)
    ASSEMBLE = "assemble"  # Make hearts at assembler
    DELIVER = "deliver"  # Deposit hearts to chest
    RECHARGE = "recharge"  # Recharge energy at charger
    CRAFT_UNCLIP = "craft_unclip"  # Craft unclip items at assembler
    UNCLIP = "unclip"  # Unclip extractors


@dataclass
class ExtractorInfo:
    """Tracks a discovered extractor with full state."""

    position: tuple[int, int]
    resource_type: str  # "carbon", "oxygen", "germanium", "silicon"
    last_seen_step: int
    times_used: int = 0

    # Extractor state from observations
    converting: bool = False  # Is it currently converting?
    cooldown_remaining: int = 0  # Steps until ready
    clipped: bool = False  # Is it depleted?
    remaining_uses: int = 999  # How many uses left


@dataclass
class ObjectState:
    """State of a single object at a position."""

    name: str

    # Extractor/station features
    converting: int = 0
    cooldown_remaining: int = 0
    clipped: int = 0
    remaining_uses: int = 999

    # Protocol details (recipes for assemblers/extractors)
    protocol_inputs: dict[str, int] = field(default_factory=dict)
    protocol_outputs: dict[str, int] = field(default_factory=dict)

    # Agent features (when object is another agent)
    agent_id: int = -1  # Which agent (-1 if not an agent) - NOT in observations, kept for API
    agent_group: int = -1  # Team/group
    agent_frozen: int = 0  # Is frozen?
    agent_orientation: int = 0  # Direction facing (0=N, 1=E, 2=S, 3=W)
    agent_visitation_counts: int = 0  # How many times visited (for tracking)


@dataclass
class ParsedObservation:
    """Parsed observation data in a clean format."""

    # Agent state
    row: int
    col: int
    energy: int

    # Inventory
    carbon: int
    oxygen: int
    germanium: int
    silicon: int
    hearts: int
    decoder: int
    modulator: int
    resonator: int
    scrambler: int

    # Nearby objects with full state (position -> ObjectState)
    nearby_objects: dict[tuple[int, int], ObjectState]


@dataclass
class SimpleAgentState:
    """State for a single agent."""

    agent_id: int

    phase: Phase = Phase.GATHER  # Start gathering immediately
    step_count: int = 0

    # Current position (origin-relative, starting at (0, 0))
    row: int = 0
    col: int = 0
    energy: int = 100

    # Per-agent discovered extractors and stations (no shared state, each agent tracks independently)
    extractors: dict[str, list[ExtractorInfo]] = field(
        default_factory=lambda: {"carbon": [], "oxygen": [], "germanium": [], "silicon": []}
    )
    stations: dict[str, tuple[int, int] | None] = field(
        default_factory=lambda: {"assembler": None, "chest": None, "charger": None}
    )

    # Inventory
    carbon: int = 0
    oxygen: int = 0
    germanium: int = 0
    silicon: int = 0
    hearts: int = 0
    decoder: int = 0
    modulator: int = 0
    resonator: int = 0
    scrambler: int = 0

    # Current target
    target_position: Optional[tuple[int, int]] = None
    target_resource: Optional[str] = None

    # Map knowledge
    map_height: int = 0
    map_width: int = 0
    occupancy: list[list[int]] = field(default_factory=list)  # 1=free, 2=obstacle (initialized in reset)

    # Track last action for position updates
    last_action: Action = field(default_factory=lambda: Action(name="noop"))

    # Current glyph (vibe) for interacting with assembler
    current_glyph: str = "default"

    # Discovered assembler recipe (dynamically discovered from observations)
    heart_recipe: Optional[dict[str, int]] = None

    # Extractor activation state
    waiting_at_extractor: Optional[tuple[int, int]] = None
    wait_steps: int = 0
    pending_use_resource: Optional[str] = None
    pending_use_amount: int = 0

    # Random walk state (for exploration within GATHER phase)
    explore_direction: int = -1

    # Frontier exploration state (tracking visited cells)
    visited_map: Optional[list[list[int]]] = None
    exploration_target: Optional[tuple[int, int]] = None  # Commit to exploration target
    exploration_target_step: int = 0  # When we set the target

    # Track unreachable extractors (position -> consecutive failed pathfind attempts)
    unreachable_extractors: dict[tuple[int, int], int] = field(default_factory=dict)

    # Agent positions (cleared and refreshed every step)
    agent_occupancy: set[tuple[int, int]] = field(default_factory=set)

    # Stuck detection
    last_position: Optional[tuple[int, int]] = None
    stuck_counter: int = 0
    position_history: list[tuple[int, int]] = field(default_factory=list)  # Last 10 positions
    stuck_loop_detected: bool = False
    stuck_escape_target: Optional[tuple[int, int]] = None
    stuck_escape_step: int = 0

    # Agent position update frequency (to avoid constant re-planning)
    agent_positions_last_updated: int = 0

    # Path caching for efficient navigation (per-agent)
    cached_path: Optional[list[tuple[int, int]]] = None
    cached_path_target: Optional[tuple[int, int]] = None
    cached_path_reach_adjacent: bool = False


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

        # Debug logging (can be enabled externally)
        self._debug = True
        self._debug_file = None

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
            "converting",
            "cooldown_remaining",
            "clipped",
            "remaining_uses",
        }
        agent_feature_pairs = {
            "agent:group": "agent_group",
            "agent:frozen": "agent_frozen",
            "agent:orientation": "agent_orientation",
            "agent:visitation_counts": "agent_visitation_counts",
        }
        self._agent_feature_key_by_name: dict[str, str] = agent_feature_pairs

        # Protocol feature prefixes (for dynamic recipe discovery)
        self._protocol_input_prefix = "protocol_input:"
        self._protocol_output_prefix = "protocol_output:"

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

    def _process_feature_at_position(
        self,
        position_features: dict[tuple[int, int], dict[str, Union[int, list[int], dict[str, int]]]],
        pos: tuple[int, int],
        feature_name: str,
        value: int,
    ) -> None:
        """Process a single observation feature and add it to position_features."""
        if pos not in position_features:
            position_features[pos] = {}

        # Handle spatial features (tag, converting, cooldown, etc.)
        if feature_name in self._spatial_feature_names:
            # Tag: collect all tags as a list (objects can have multiple tags)
            if feature_name == "tag":
                tags = position_features[pos].setdefault("tags", [])
                if isinstance(tags, list):
                    tags.append(value)
                return
            # Other spatial features are single values
            position_features[pos][feature_name] = value
            return

        # Handle agent features (agent:group -> agent_group, etc.)
        agent_feature_key = self._agent_feature_key_by_name.get(feature_name)
        if agent_feature_key is not None:
            position_features[pos][agent_feature_key] = value
            return

        # Handle protocol features (recipes)
        if feature_name.startswith(self._protocol_input_prefix):
            resource = feature_name[len(self._protocol_input_prefix) :]
            inputs = position_features[pos].setdefault("protocol_inputs", {})
            if isinstance(inputs, dict):
                inputs[resource] = value
            return

        if feature_name.startswith(self._protocol_output_prefix):
            resource = feature_name[len(self._protocol_output_prefix) :]
            outputs = position_features[pos].setdefault("protocol_outputs", {})
            if isinstance(outputs, dict):
                outputs[resource] = value
            return

    def _create_object_state(self, features: dict[str, Union[int, list[int], dict[str, int]]]) -> ObjectState:
        """Create an ObjectState from collected features.

        Note: Objects can have multiple tags (e.g., "wall" + "green" vibe).
        We use the first tag as the primary object name.
        """
        # Get tags list (now stored as "tags" instead of "tag")
        tags_value = features.get("tags", [])
        if isinstance(tags_value, list):
            tag_ids = list(tags_value)
        elif isinstance(tags_value, int):
            tag_ids = [tags_value]
        else:
            tag_ids = []

        # Use first tag as primary object name
        if tag_ids:
            primary_tag_id = tag_ids[0]
            obj_name = self._tag_names.get(primary_tag_id, f"unknown_tag_{primary_tag_id}")
        else:
            obj_name = "unknown"

        # Helper to safely extract int values
        def get_int(key: str, default: int) -> int:
            val = features.get(key, default)
            return int(val) if isinstance(val, int) else default

        # Helper to safely extract dict values
        def get_dict(key: str) -> dict[str, int]:
            val = features.get(key, {})
            return dict(val) if isinstance(val, dict) else {}

        return ObjectState(
            name=obj_name,
            converting=get_int("converting", 0),
            cooldown_remaining=get_int("cooldown_remaining", 0),
            clipped=get_int("clipped", 0),
            remaining_uses=get_int("remaining_uses", 999),
            protocol_inputs=get_dict("protocol_inputs"),
            protocol_outputs=get_dict("protocol_outputs"),
            agent_group=get_int("agent_group", -1),
            agent_frozen=get_int("agent_frozen", 0),
            agent_orientation=get_int("agent_orientation", 0),
        )

    def parse_observation(
        self, state: SimpleAgentState, obs: AgentObservation, debug: bool = False
    ) -> ParsedObservation:
        """Parse token-based observation into structured format.

        AgentObservation with tokens (ObservationToken list)
        - Inventory is obtained via agent.inventory (not parsed here)
        - Only spatial features are parsed from observations

        Converts egocentric spatial coordinates to world coordinates using agent position.
        Agent position (agent_row, agent_col) comes from simulation.grid_objects().
        """
        # First pass: collect all spatial features by position
        position_features: dict[tuple[int, int], dict[str, Union[int, list[int], dict[str, int]]]] = {}

        for tok in obs.tokens:
            obs_r, obs_c = tok.location
            feature_name = tok.feature.name
            value = tok.value

            # Skip center location - that's inventory/global obs, obtained via agent.inventory
            if obs_r == self._obs_hr and obs_c == self._obs_wr:
                continue

            # Convert observation-relative coords to world coords
            if state.row >= 0 and state.col >= 0:
                r = obs_r - self._obs_hr + state.row
                c = obs_c - self._obs_wr + state.col

                if 0 <= r < state.map_height and 0 <= c < state.map_width:
                    self._process_feature_at_position(position_features, (r, c), feature_name, value)

        # Second pass: create ObjectState for each position with tags
        nearby_objects = {
            pos: self._create_object_state(features)
            for pos, features in position_features.items()
            if "tags" in features  # Note: stored as "tags" (plural) to support multiple tags per object
        }

        return ParsedObservation(
            row=state.row,
            col=state.col,
            energy=0,  # Inventory obtained via agent.inventory
            carbon=0,
            oxygen=0,
            germanium=0,
            silicon=0,
            hearts=0,
            decoder=0,
            modulator=0,
            resonator=0,
            scrambler=0,
            nearby_objects=nearby_objects,
        )

    def _try_find_escape_target(self, s: SimpleAgentState) -> Optional[tuple[int, int]]:
        """Try to find a distant free cell for escape. Returns None if not found."""
        for _ in range(50):  # Try 50 random locations
            rand_r = random.randint(0, s.map_height - 1)
            rand_c = random.randint(0, s.map_width - 1)
            dist = abs(rand_r - s.row) + abs(rand_c - s.col)
            if dist >= self._hyperparams.stuck_escape_distance and s.occupancy[rand_r][rand_c] == CellType.FREE.value:
                return (rand_r, rand_c)
        return None

    def _try_random_direction(self, s: SimpleAgentState) -> Optional[Action]:
        """Try to move in any free adjacent direction. Returns None if all blocked."""
        directions: list[CardinalDirection] = ["north", "south", "east", "west"]
        random.shuffle(directions)
        for direction in directions:
            dr, dc = self._move_deltas[direction]
            nr, nc = s.row + dr, s.col + dc
            if self._is_within_bounds(s, nr, nc) and s.occupancy[nr][nc] == CellType.FREE.value:
                return self._actions.move.Move(direction)
        return None

    def _clear_stuck_state(self, s: SimpleAgentState) -> None:
        """Clear all stuck detection state."""
        s.stuck_loop_detected = False
        s.stuck_escape_target = None

    def _check_stuck_and_escape(self, s: SimpleAgentState) -> Optional[Action]:
        """Check if agent is stuck in a loop and return escape action if needed."""
        if not self._hyperparams.stuck_detection_enabled or not s.stuck_loop_detected:
            return None

        # Timeout: give up after 10 steps and resume normal behavior
        if s.step_count - s.stuck_escape_step > 10:
            self._clear_stuck_state(s)
            return None

        # If no escape target yet, try to find one
        if s.stuck_escape_target is None:
            s.stuck_escape_target = self._try_find_escape_target(s)
            if s.stuck_escape_target is not None:
                # Clear cached path to force new pathfinding
                s.cached_path = None
                s.cached_path_target = None
            else:
                # Couldn't find distant target, try any adjacent move
                action = self._try_random_direction(s)
                self._clear_stuck_state(s)
                return action if action else self._actions.noop.Noop()

        # Have escape target: move toward it
        if (s.row, s.col) == s.stuck_escape_target:
            # Reached target, resume normal behavior
            self._clear_stuck_state(s)
            return None

        return self._move_towards(s, s.stuck_escape_target)

    def _last_action_moved_agent(self, state: SimpleAgentState) -> bool:
        """
        Determine if the last action actually moved the agent's position.

        CRITICAL INSIGHT: When using objects (extractors, assemblers, chargers, chests),
        the agent issues a "move" action to activate them, but the agent DOESN'T actually move!
        The move action is just the mechanism for using the object.

        The agent should NEVER try to move into obstacles (walls). The pathfinding should
        prevent that. So if a move action was issued, it was either:
        1. A move to an empty cell (agent moves) - return True
        2. A move to use an object (agent doesn't move) - return False

        We can distinguish these by checking if the destination is an obstacle in the occupancy map.
        """
        # If no last action or not a move action, no position change
        if state.last_action is None or not state.last_action.name.startswith("move_"):
            return False

        # If occupancy map not initialized yet, assume move succeeded
        if not state.occupancy or state.row < 0:
            return True

        # Extract direction from last action
        direction = state.last_action.name[5:]  # Remove "move_" prefix
        if direction not in self._move_deltas:
            return False

        # Calculate the destination cell (where the move action was directed)
        dr, dc = self._move_deltas[direction]
        dest_row = state.row + dr
        dest_col = state.col + dc

        # Check if destination is out of bounds - agent didn't move
        if not (0 <= dest_row < state.map_height and 0 <= dest_col < state.map_width):
            if self._debug:
                msg = f"[Agent {state.agent_id}] Move out of bounds - no position change\n"
                if hasattr(self, "_debug_file") and self._debug_file:
                    self._debug_file.write(msg)
                    self._debug_file.flush()
            return False

        # Check if the destination is an obstacle (extractor, assembler, wall, etc.)
        if state.occupancy[dest_row][dest_col] == CellType.OBSTACLE.value:
            # Destination is an obstacle - this was a "move to use object" action
            # Agent didn't actually move
            if self._debug:
                msg = f"[Agent {state.agent_id}] Move to use object at ({dest_row},{dest_col}) - no position change\n"
                if hasattr(self, "_debug_file") and self._debug_file:
                    self._debug_file.write(msg)
                    self._debug_file.flush()
            return False

        # Check if destination has another agent - agent didn't move (blocked)
        if (dest_row, dest_col) in state.agent_occupancy:
            if self._debug:
                msg = (
                    f"[Agent {state.agent_id}] Move blocked by another agent at "
                    f"({dest_row},{dest_col}) - no position change\n"
                )
                if hasattr(self, "_debug_file") and self._debug_file:
                    self._debug_file.write(msg)
                    self._debug_file.flush()
            return False

        # Destination was free - agent actually moved!
        return True

    def step_with_state(self, obs: AgentObservation, state: SimpleAgentState) -> Tuple[Action, SimpleAgentState]:
        """Compute action for one agent (returns action index)."""
        state.step_count += 1

        state.agent_occupancy.clear()

        # CRITICAL FIX: Parse observation FIRST to update occupancy map with CURRENT data
        # The old code used stale occupancy data from the previous step, causing position drift!
        # Now we:
        # 1. Parse current observation to get current occupancy map
        # 2. Use current occupancy to determine if last action moved us
        # 3. Update position based on that determination

        # Parse observation and update occupancy map (but don't update position yet)
        parsed = self.parse_observation(state, obs)

        # Read inventory from observation tokens at center cell
        inv = {}
        center_r, center_c = self._obs_hr, self._obs_wr
        for tok in obs.tokens:
            if tok.location == (center_r, center_c):
                feature_name = tok.feature.name
                if feature_name.startswith("inv:"):
                    resource_name = feature_name[4:]  # Remove "inv:" prefix
                    inv[resource_name] = tok.value

        state.energy = inv.get("energy", 0)
        state.carbon = inv.get("carbon", 0)
        state.oxygen = inv.get("oxygen", 0)
        state.germanium = inv.get("germanium", 0)
        state.silicon = inv.get("silicon", 0)
        state.hearts = inv.get("heart", 0)
        state.decoder = inv.get("decoder", 0)
        state.modulator = inv.get("modulator", 0)
        state.resonator = inv.get("resonator", 0)
        state.scrambler = inv.get("scrambler", 0)

        self._update_occupancy_and_discover(state, parsed)

        # NOW check if last action moved us, using CURRENT occupancy map
        action_moved_agent = self._last_action_moved_agent(state)

        # Update position based on whether last action actually moved us
        self._update_agent_position(state, action_moved_agent)

        # Check if we received resources from pending extractor use
        self._check_pending_extractor_use(state)

        self._update_phase(state)

        # Debug: Log phase and energy
        if self._debug:
            msg = f"[Agent {state.agent_id}] Step {state.step_count}: Phase={state.phase.name}, Energy={state.energy}\n"
            if hasattr(self, "_debug_file") and self._debug_file:
                self._debug_file.write(msg)
                self._debug_file.flush()
            else:
                print(msg, end="")

        # Update vibe to match phase
        desired_vibe = self._get_vibe_for_phase(state.phase, state)
        if state.current_glyph != desired_vibe:
            if self._debug:
                msg = (
                    f"[Agent {state.agent_id}] Changing vibe: {state.current_glyph} -> {desired_vibe} "
                    f"(target_resource={state.target_resource})\n"
                )
                if hasattr(self, "_debug_file") and self._debug_file:
                    self._debug_file.write(msg)
                    self._debug_file.flush()
            state.current_glyph = desired_vibe
            # Return vibe change action this step
            action = self._actions.change_vibe.ChangeVibe(VIBE_BY_NAME[desired_vibe])
            state.last_action = action
            return action, state

        # Check for stuck loop and attempt escape
        action = self._check_stuck_and_escape(state)
        if action is not None:
            state.last_action = action
            return action, state

        action = self._execute_phase(state)

        # Save action for next step's position update
        state.last_action = action

        if self._debug:
            # Comprehensive debug output
            msg = f"\n{'=' * 80}\n"
            msg += f"[Agent {state.agent_id}] Step {state.step_count}\n"
            msg += f"  Phase: {state.phase.name}\n"
            msg += f"  Agent believed position: ({state.row}, {state.col})\n"
            msg += f"  Energy: {state.energy}\n"
            msg += (
                f"  Inventory: carbon={state.carbon}, oxygen={state.oxygen}, "
                f"germanium={state.germanium}, silicon={state.silicon}, hearts={state.hearts}\n"
            )
            msg += f"  Target resource: {state.target_resource}\n"
            msg += f"  Pending use: {state.pending_use_resource}\n"
            msg += f"  Heart recipe: {state.heart_recipe}\n"

            # Show discovered extractors and distances
            if state.extractors:
                msg += "  Discovered extractors:\n"
                for res_type, extractors in state.extractors.items():
                    for ext in extractors:
                        dist = abs(ext.position[0] - state.row) + abs(ext.position[1] - state.col)
                        msg += (
                            f"    {res_type} at {ext.position}: dist={dist}, uses={ext.remaining_uses}, "
                            f"cooldown={ext.cooldown_remaining}, clipped={ext.clipped}\n"
                        )

            # Show discovered stations and distances
            if any(state.stations.values()):
                msg += "  Discovered stations:\n"
                for station_type, pos in state.stations.items():
                    if pos:
                        dist = abs(pos[0] - state.row) + abs(pos[1] - state.col)
                        msg += f"    {station_type} at {pos}: dist={dist}\n"

            msg += f"  Chosen action: {action.name}\n"
            msg += f"{'=' * 80}\n"

            if hasattr(self, "_debug_file") and self._debug_file:
                self._debug_file.write(msg)
                self._debug_file.flush()
            else:
                print(msg, end="")

        return action, state

    def _update_inventory_from_parsed(self, s: SimpleAgentState, parsed: ParsedObservation) -> None:
        """Update inventory from parsed observation."""
        s.energy = parsed.energy
        s.carbon = parsed.carbon
        s.oxygen = parsed.oxygen
        s.germanium = parsed.germanium
        s.silicon = parsed.silicon
        s.hearts = parsed.hearts
        s.decoder = parsed.decoder
        s.modulator = parsed.modulator
        s.resonator = parsed.resonator
        s.scrambler = parsed.scrambler

    def _check_pending_extractor_use(self, s: SimpleAgentState) -> None:
        """Check if we received resources from a pending extractor use."""
        if s.pending_use_resource is not None:
            current_amount = getattr(s, s.pending_use_resource, 0)
            if current_amount > s.pending_use_amount:
                # Extractor gave us the resource!
                s.pending_use_resource = None
                s.pending_use_amount = 0
                s.waiting_at_extractor = None
                s.wait_steps = 0

    def _update_occupancy_and_discover(self, s: SimpleAgentState, parsed: ParsedObservation) -> None:
        """Update occupancy map and discover objects from parsed observation."""
        # Discover heart recipe from assembler protocol (if not yet discovered)
        if s.heart_recipe is None:
            for _pos, obj_state in parsed.nearby_objects.items():
                if obj_state.name == "assembler" and obj_state.protocol_inputs:
                    # Check if this is the heart recipe (outputs "heart")
                    if obj_state.protocol_outputs.get("heart", 0) > 0:
                        s.heart_recipe = dict(obj_state.protocol_inputs)
                        if DEBUG:
                            print(f"[Agent {s.agent_id}] Discovered heart recipe: {s.heart_recipe}")
                        break

        # Update occupancy map and discover extractors/stations
        self._discover_objects(s, parsed)

    def _update_agent_position(self, s: SimpleAgentState, action_success: bool) -> None:
        """Update agent position based on last action and whether it succeeded.

        Position is tracked relative to origin (starting position), using only movement deltas.
        No dependency on simulation.grid_objects().
        """
        # If last action was a move and it succeeded, update position
        if action_success and s.last_action.name.startswith("move_"):
            # Extract direction from action name (e.g., "move_north" -> "north")
            direction = s.last_action.name[5:]  # Remove "move_" prefix
            if direction in self._move_deltas:
                dr, dc = self._move_deltas[direction]
                old_pos = (s.row, s.col)
                s.row += dr
                s.col += dc

                if self._debug:
                    msg = f"[Agent {s.agent_id}] Position update: {old_pos} -> ({s.row},{s.col}) via {direction}\n"
                    if hasattr(self, "_debug_file") and self._debug_file:
                        self._debug_file.write(msg)
                        self._debug_file.flush()
                    else:
                        print(msg, end="")

        # Update position history and detect loops
        current_pos = (s.row, s.col)
        s.position_history.append(current_pos)
        if len(s.position_history) > 10:
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

    def _update_state_from_obs(self, s: SimpleAgentState, obs: AgentObservation, action_success: bool = True) -> None:
        """Update agent state from observation."""

        # STEP 1: Update agent position based on last action (origin-relative positioning)
        self._update_agent_position(s, action_success)

        inv = {}
        center_r, center_c = self._obs_hr, self._obs_wr  # Center of egocentric observation

        for tok in obs.tokens:
            # Only read inventory from the center cell (where this agent is)
            if tok.location == (center_r, center_c):
                feature_spec = tok.feature
                feature_name = feature_spec.name
                if feature_name.startswith("inv:"):
                    resource_name = feature_name[4:]  # Remove "inv:" prefix
                    inv[resource_name] = tok.value

        s.energy = inv.get("energy", 0)
        s.carbon = inv.get("carbon", 0)
        s.oxygen = inv.get("oxygen", 0)
        s.germanium = inv.get("germanium", 0)
        s.silicon = inv.get("silicon", 0)
        s.hearts = inv.get("heart", 0)
        s.decoder = inv.get("decoder", 0)
        s.modulator = inv.get("modulator", 0)
        s.resonator = inv.get("resonator", 0)
        s.scrambler = inv.get("scrambler", 0)

        # STEP 3: Parse spatial observation with known position
        debug = s.step_count <= 2
        parsed = self.parse_observation(s, obs, debug=debug)

        # STEP 3.5: Discover heart recipe from assembler protocol (if not yet discovered)
        if s.heart_recipe is None:
            for _pos, obj_state in parsed.nearby_objects.items():
                if obj_state.name == "assembler" and obj_state.protocol_inputs:
                    # Check if this is the heart recipe (outputs "heart")
                    if obj_state.protocol_outputs.get("heart", 0) > 0:
                        s.heart_recipe = dict(obj_state.protocol_inputs)
                        if DEBUG:
                            print(f"[Agent {s.agent_id}] Discovered heart recipe: {s.heart_recipe}")
                        break

        # Check if we received the resource from an extractor activation
        if s.pending_use_resource is not None:
            current_amount = getattr(s, s.pending_use_resource, 0)
            # Extractor gave us the resource!
            if current_amount > s.pending_use_amount:
                s.pending_use_resource = None
                s.pending_use_amount = 0
                s.waiting_at_extractor = None
                s.wait_steps = 0
            # Still waiting for extractor to finish converting - this is normal
            else:
                pass  # Keep waiting

        # STEP 4: Discover objects (updates occupancy map)
        self._discover_objects(s, parsed)

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
            if self._is_wall(obj_name):
                s.occupancy[r][c] = CellType.OBSTACLE.value
                continue

            # Other agents: track their positions but don't mark as obstacles
            if obj_name == "agent" and obj_state.agent_id != s.agent_id:
                s.agent_occupancy.add((r, c))
                continue

            # Discover stations (all stations are obstacles - can't walk through them)
            for station_name in ("assembler", "chest", "charger"):
                if self._is_station(obj_name, station_name):
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

    def _is_wall(self, obj_name: str) -> bool:
        return "wall" in obj_name or "#" in obj_name or obj_name in {"wall", "obstacle"}

    def _is_floor(self, obj_name: str) -> bool:
        # environment returns empty string for empty cells
        return obj_name in {"floor", ""}

    def _is_station(self, obj_name: str, station: str) -> bool:
        return station in obj_name

    def _discover_station(self, s: SimpleAgentState, pos: tuple[int, int], station_key: str) -> None:
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

            if self._debug:
                # Calculate what observation coordinate this came from
                obs_r = pos[0] - s.row + self._obs_hr
                obs_c = pos[1] - s.col + self._obs_wr
                msg = (
                    f"[Agent {s.agent_id}] NEW {resource_type} at world_pos={pos}, "
                    f"agent_pos=({s.row},{s.col}), obs_coord=({obs_r},{obs_c})\n"
                    f"  uses={extractor.remaining_uses}, clipped={extractor.clipped}, "
                    f"cooldown={extractor.cooldown_remaining}\n"
                )
                if hasattr(self, "_debug_file") and self._debug_file:
                    self._debug_file.write(msg)
                    self._debug_file.flush()
                else:
                    print(msg, end="")

        extractor.last_seen_step = s.step_count
        extractor.converting = obj_state.converting > 0
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
                if self._debug:
                    msg = (
                        f"[Agent {s.agent_id}] Exiting RECHARGE: energy={s.energy} >= "
                        f"threshold={self._hyperparams.recharge_threshold_high}\n"
                    )
                    if hasattr(self, "_debug_file") and self._debug_file:
                        self._debug_file.write(msg)
                        self._debug_file.flush()
                s.phase = Phase.GATHER
                s.target_position = None
            else:
                if self._debug:
                    msg = (
                        f"[Agent {s.agent_id}] Staying in RECHARGE: energy={s.energy} < "
                        f"threshold={self._hyperparams.recharge_threshold_high}\n"
                    )
                    if hasattr(self, "_debug_file") and self._debug_file:
                        self._debug_file.write(msg)
                        self._debug_file.flush()
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

        # Priority 5: Default to GATHER
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
            return state.target_resource

        phase_to_vibe = {
            Phase.GATHER: "carbon",  # Default fallback if no target resource
            Phase.ASSEMBLE: "heart",  # Red for assembly
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

    def _explore_frontier(self, s: SimpleAgentState) -> Action:
        """
        Frontier-based exploration: move toward areas we haven't visited recently.
        Uses a 'visited' tracking system to encourage exploring new areas.
        More efficient than random walk for systematic map coverage.

        Includes occasional random moves to break out of repetitive patterns.
        """
        if s.row < 0:
            return self._actions.noop.Noop()

        # Initialize visited map if not exists
        if s.visited_map is None:
            s.visited_map = [[0 for _ in range(s.map_width)] for _ in range(s.map_height)]

        # Mark current cell as visited
        s.visited_map[s.row][s.col] = s.step_count

        # 10% chance to take a random move to break patterns
        if random.random() < 0.1:
            # Pick a random valid direction
            valid_moves = []
            for action, (dr, dc) in [
                (self._actions.move.Move("north"), (-1, 0)),
                (self._actions.move.Move("south"), (1, 0)),
                (self._actions.move.Move("east"), (0, 1)),
                (self._actions.move.Move("west"), (0, -1)),
            ]:
                nr, nc = s.row + dr, s.col + dc
                if self._is_within_bounds(s, nr, nc) and s.occupancy[nr][nc] == CellType.FREE.value:
                    valid_moves.append(action)

            if valid_moves:
                # Clear exploration target when taking random move
                s.exploration_target = None
                return random.choice(valid_moves)

        # Check if we should keep current exploration target
        if s.exploration_target is not None:
            tr, tc = s.exploration_target
            # Keep target if: we set it recently AND haven't reached it yet
            if (
                s.step_count - s.exploration_target_step < self._hyperparams.exploration_target_persistence
                and (s.row, s.col) != s.exploration_target
            ):
                # Check if target is still valid (FREE and in bounds)
                if self._is_within_bounds(s, tr, tc) and s.occupancy[tr][tc] == CellType.FREE.value:
                    return self._move_towards(s, s.exploration_target)
            # Target reached or expired, clear it
            s.exploration_target = None

        # Calculate which quadrants are least explored
        # This helps bias exploration toward neglected areas of the map
        map_center_r, map_center_c = s.map_height // 2, s.map_width // 2

        # Determine which quadrant we're in and which is opposite
        in_top = s.row < map_center_r
        in_left = s.col < map_center_c

        # Bonus for exploring opposite quadrant (helps break out of local exploration)
        def get_quadrant_bonus(r, c):
            target_top = r < map_center_r
            target_left = c < map_center_c
            # Give bonus if target is in opposite quadrant
            if in_top != target_top or in_left != target_left:
                return self._hyperparams.exploration_quadrant_bonus
            return 0

        # Find nearest least-recently-visited cell
        best_target = None
        best_score = -float("inf")

        # Sample cells in expanding radius - search much further to find unexplored regions
        max_radius = max(s.map_height, s.map_width) // 2 + 5
        for radius in range(5, max_radius, 2):  # Step by 2 for efficiency
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if abs(dr) != radius and abs(dc) != radius:
                        continue  # Only check perimeter of current radius

                    r, c = s.row + dr, s.col + dc
                    if not self._is_within_bounds(s, r, c):
                        continue

                    if s.occupancy[r][c] != CellType.FREE.value:
                        continue

                    # Score based on how long ago we visited (higher = less recently visited)
                    last_visited = s.visited_map[r][c]
                    visit_score = s.step_count - last_visited
                    distance = abs(dr) + abs(dc)

                    # Visit weight and distance penalty control exploration behavior
                    # This helps agents balance between exploring far regions vs nearby cells
                    quadrant_bonus = get_quadrant_bonus(r, c)
                    score = (
                        visit_score * self._hyperparams.exploration_visit_weight
                        - (distance * self._hyperparams.exploration_distance_penalty)
                        + quadrant_bonus
                    )

                    if score > best_score:
                        best_score = score
                        best_target = (r, c)

            # Only break early if we found an excellent target (very high threshold)
            # This ensures we search wide enough to find truly unexplored areas
            if best_target and best_score > 800:
                break

        if best_target:
            # Commit to this target for multiple steps
            s.exploration_target = best_target
            s.exploration_target_step = s.step_count
            return self._move_towards(s, best_target)

        # No good target found, pick a random direction
        return self._actions.move.Move(random.choice(CardinalDirections))

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

        return self._explore(s)

    def _explore(self, s: SimpleAgentState) -> Action:
        """Execute exploration using frontier-based strategy."""
        return self._explore_frontier(s)

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
        if s.pending_use_resource is None:
            return None

        if self._debug:
            msg = (
                f"[Agent {s.agent_id}] _handle_waiting_for_extractor: waiting for {s.pending_use_resource}\n"
                f"  wait_steps={s.wait_steps}, waiting_at={s.waiting_at_extractor}\n"
            )
            if hasattr(self, "_debug_file") and self._debug_file:
                self._debug_file.write(msg)
                self._debug_file.flush()

        # Look up the extractor we're waiting for
        extractor = self._find_extractor_at_position(s, s.waiting_at_extractor)

        # Calculate timeout based on observed cooldown
        max_wait = extractor.cooldown_remaining + 5 if extractor else 20

        s.wait_steps += 1
        if s.wait_steps > max_wait:
            if self._debug:
                msg = "  → Timeout! Clearing pending_use state\n"
                if hasattr(self, "_debug_file") and self._debug_file:
                    self._debug_file.write(msg)
                    self._debug_file.flush()
            # Timeout - reset and try again
            s.pending_use_resource = None
            s.pending_use_amount = 0
            self._clear_waiting_state(s)

        if self._debug:
            msg = "  → Returning noop (still waiting)\n"
            if hasattr(self, "_debug_file") and self._debug_file:
                self._debug_file.write(msg)
                self._debug_file.flush()

        return self._actions.noop.Noop()

    def _navigate_to_extractor(
        self, s: SimpleAgentState, extractor: ExtractorInfo, resource_type: str
    ) -> Optional[Action]:
        """Navigate to extractor. Returns action if navigating, None if already adjacent."""
        er, ec = extractor.position
        dr = abs(s.row - er)
        dc = abs(s.col - ec)
        is_adjacent = (dr == 1 and dc == 0) or (dr == 0 and dc == 1)

        if self._debug:
            msg = (
                f"[Agent {s.agent_id}] _navigate_to_extractor: {resource_type} at {extractor.position}\n"
                f"  Agent at ({s.row},{s.col}), extractor at ({er},{ec})\n"
                f"  Distance: dr={dr}, dc={dc}, is_adjacent={is_adjacent}\n"
            )
            if hasattr(self, "_debug_file") and self._debug_file:
                self._debug_file.write(msg)
                self._debug_file.flush()

        if is_adjacent:
            if self._debug:
                msg = "  → Already adjacent, proceeding to use\n"
                if hasattr(self, "_debug_file") and self._debug_file:
                    self._debug_file.write(msg)
                    self._debug_file.flush()
            return None  # Already adjacent

        # Move towards extractor
        if self._debug:
            msg = "  → Not adjacent, navigating towards extractor\n"
            if hasattr(self, "_debug_file") and self._debug_file:
                self._debug_file.write(msg)
                self._debug_file.flush()
        self._clear_waiting_state(s)
        action = self._move_towards(s, extractor.position, reach_adjacent=True)
        if action == self._actions.noop.Noop():
            return self._explore(s)
        return action

    def _use_extractor_if_ready(self, s: SimpleAgentState, extractor: ExtractorInfo, resource_type: str) -> Action:
        """Try to use extractor if ready. Returns appropriate action."""
        if self._debug:
            msg = (
                f"[Agent {s.agent_id}] _use_extractor_if_ready: {resource_type} at {extractor.position}\n"
                f"  Agent at ({s.row},{s.col}), cooldown={extractor.cooldown_remaining}, "
                f"uses={extractor.remaining_uses}, clipped={extractor.clipped}\n"
            )
            if hasattr(self, "_debug_file") and self._debug_file:
                self._debug_file.write(msg)
                self._debug_file.flush()

        # Wait if on cooldown
        if extractor.cooldown_remaining > 0 or extractor.converting:
            s.waiting_at_extractor = extractor.position
            s.wait_steps += 1
            if self._debug:
                msg = "  → Waiting for cooldown/conversion (noop)\n"
                if hasattr(self, "_debug_file") and self._debug_file:
                    self._debug_file.write(msg)
                    self._debug_file.flush()
            return self._actions.noop.Noop()

        # Skip if depleted/clipped
        if extractor.remaining_uses == 0 or extractor.clipped:
            self._clear_waiting_state(s)
            if self._debug:
                msg = "  → Extractor depleted/clipped (noop)\n"
                if hasattr(self, "_debug_file") and self._debug_file:
                    self._debug_file.write(msg)
                    self._debug_file.flush()
            return self._actions.noop.Noop()

        # Use it! Track pre-use inventory and activate
        old_amount = getattr(s, resource_type, 0)
        action = self._move_into_cell(s, extractor.position)

        # Set waiting state to detect when resource is received
        s.pending_use_resource = resource_type
        s.pending_use_amount = old_amount
        s.waiting_at_extractor = extractor.position

        if self._debug:
            msg = f"  → Using extractor! Action: {action.name}\n"
            if hasattr(self, "_debug_file") and self._debug_file:
                self._debug_file.write(msg)
                self._debug_file.flush()

        return action

    def _do_gather(self, s: SimpleAgentState) -> Action:
        """
        Gather resources from nearest extractors.
        Opportunistically uses ANY extractor for ANY needed resource.
        """
        if self._debug:
            msg = f"[Agent {s.agent_id}] _do_gather called: pos=({s.row},{s.col})\n"
            msg += f"  Extractors discovered: {sum(len(exts) for exts in s.extractors.values())}\n"
            for res_type, exts in s.extractors.items():
                if exts:
                    msg += f"    {res_type}: {[e.position for e in exts]}\n"
            if hasattr(self, "_debug_file") and self._debug_file:
                self._debug_file.write(msg)
                self._debug_file.flush()
            else:
                print(msg, end="")

        # Handle waiting for activated extractor
        wait_action = self._handle_waiting_for_extractor(s)
        if wait_action is not None:
            if self._debug:
                msg = f"[Agent {s.agent_id}] Waiting for extractor, returning wait action\n"
                if hasattr(self, "_debug_file") and self._debug_file:
                    self._debug_file.write(msg)
                    self._debug_file.flush()
            return wait_action

        # Check resource deficits
        deficits = self._calculate_deficits(s)
        if self._debug:
            msg = f"[Agent {s.agent_id}] Deficits: {deficits}\n"
            if hasattr(self, "_debug_file") and self._debug_file:
                self._debug_file.write(msg)
                self._debug_file.flush()

        if all(d <= 0 for d in deficits.values()):
            self._clear_waiting_state(s)
            if self._debug:
                msg = f"[Agent {s.agent_id}] No deficits, returning noop\n"
                if hasattr(self, "_debug_file") and self._debug_file:
                    self._debug_file.write(msg)
                    self._debug_file.flush()
            return self._actions.noop.Noop()

        # Explore until we find an extractor for a needed resource
        if self._debug:
            msg = f"[Agent {s.agent_id}] Checking if we need to explore for extractors...\n"
            if hasattr(self, "_debug_file") and self._debug_file:
                self._debug_file.write(msg)
                self._debug_file.flush()

        explore_action = self._explore_until(
            s,
            condition=lambda: self._find_any_needed_extractor(s) is not None,
            reason=f"Need extractors for: {', '.join(k for k, v in deficits.items() if v > 0)}",
        )
        if explore_action is not None:
            if self._debug:
                msg = f"[Agent {s.agent_id}] Still exploring, action: {explore_action.name}\n"
                if hasattr(self, "_debug_file") and self._debug_file:
                    self._debug_file.write(msg)
                    self._debug_file.flush()
            return explore_action

        # Found an extractor - navigate and use it
        if self._debug:
            msg = f"[Agent {s.agent_id}] Exploration complete, finding extractor to use...\n"
            if hasattr(self, "_debug_file") and self._debug_file:
                self._debug_file.write(msg)
                self._debug_file.flush()

        result = self._find_any_needed_extractor(s)
        if result is None:
            if self._debug:
                msg = f"[Agent {s.agent_id}] No extractor found despite condition passing\n"
                if hasattr(self, "_debug_file") and self._debug_file:
                    self._debug_file.write(msg)
                    self._debug_file.flush()
            return self._explore(s)  # Shouldn't happen, but be safe

        extractor, resource_type = result
        s.exploration_target = None  # Clear exploration target
        s.target_resource = resource_type

        # Navigate to extractor if not adjacent
        nav_action = self._navigate_to_extractor(s, extractor, resource_type)
        if nav_action is not None:
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

        # First, ensure we have the correct glyph (heart) for assembling
        if s.current_glyph != "heart":
            vibe_action = self._actions.change_vibe.ChangeVibe(VIBE_BY_NAME["heart"])
            s.current_glyph = "heart"
            return vibe_action

        # Assembler is known, move adjacent to it then use it
        assembler = s.stations["assembler"]
        assert assembler is not None  # Guaranteed by _explore_until above
        ar, ac = assembler
        dr = abs(s.row - ar)
        dc = abs(s.col - ac)
        is_adjacent = (dr == 1 and dc == 0) or (dr == 0 and dc == 1)

        if self._debug:
            msg = (
                f"[Agent {s.agent_id}] _do_assemble: assembler at {assembler}\n"
                f"  Agent at ({s.row},{s.col}), distance: dr={dr}, dc={dc}, adjacent={is_adjacent}\n"
            )
            if hasattr(self, "_debug_file") and self._debug_file:
                self._debug_file.write(msg)
                self._debug_file.flush()

        if is_adjacent:
            # Adjacent - move into it to use it (assembler will consume resources and give heart)
            if self._debug:
                msg = "  → Using assembler (move into it)\n"
                if hasattr(self, "_debug_file") and self._debug_file:
                    self._debug_file.write(msg)
                    self._debug_file.flush()
            return self._move_into_cell(s, assembler)

        # Not adjacent yet, move towards it
        if self._debug:
            msg = "  → Navigating to assembler\n"
            if hasattr(self, "_debug_file") and self._debug_file:
                self._debug_file.write(msg)
                self._debug_file.flush()
        return self._move_towards(s, assembler, reach_adjacent=True)

    def _do_deliver(self, s: SimpleAgentState) -> Action:
        """Deliver hearts to chest."""
        # Explore until we find chest
        explore_action = self._explore_until(s, condition=lambda: s.stations["chest"] is not None, reason="Need chest")
        if explore_action is not None:
            return explore_action

        # First, ensure we have the correct glyph (default/neutral) for chest deposit
        # - "default" vibe: DEPOSIT resources (positive values)
        # - specific resource vibes (e.g., "heart"): WITHDRAW resources (negative values)
        if s.current_glyph != "default":
            vibe_action = self._actions.change_vibe.ChangeVibe(VIBE_BY_NAME["default"])
            s.current_glyph = "default"
            return vibe_action

        # Chest is known, move adjacent to it then use it
        chest = s.stations["chest"]
        assert chest is not None  # Guaranteed by _explore_until above
        cr, cc = chest
        dr = abs(s.row - cr)
        dc = abs(s.col - cc)
        is_adjacent = (dr == 1 and dc == 0) or (dr == 0 and dc == 1)

        if is_adjacent:
            # Adjacent - move into it to deliver
            return self._move_into_cell(s, chest)

        # Not adjacent yet, move towards it
        return self._move_towards(s, chest, reach_adjacent=True)

    def _do_recharge(self, s: SimpleAgentState) -> Action:
        """Recharge at charger."""
        # Explore until we find charger
        explore_action = self._explore_until(
            s, condition=lambda: s.stations["charger"] is not None, reason="Need charger"
        )
        if explore_action is not None:
            return explore_action

        # Charger is known, move adjacent to it then use it (same as extractors/chest)
        charger = s.stations["charger"]
        assert charger is not None  # Guaranteed by _explore_until above
        chr, chc = charger
        dr = abs(s.row - chr)
        dc = abs(s.col - chc)
        is_adjacent = (dr == 1 and dc == 0) or (dr == 0 and dc == 1)

        if self._debug:
            msg = (
                f"[Agent {s.agent_id}] _do_recharge: charger at {charger}\n"
                f"  Agent at ({s.row},{s.col}), distance: dr={dr}, dc={dc}, adjacent={is_adjacent}\n"
            )
            if hasattr(self, "_debug_file") and self._debug_file:
                self._debug_file.write(msg)
                self._debug_file.flush()

        if is_adjacent:
            # Adjacent - move into it to recharge
            if self._debug:
                msg = "  → Using charger (move into it)\n"
                if hasattr(self, "_debug_file") and self._debug_file:
                    self._debug_file.write(msg)
                    self._debug_file.flush()
            return self._move_into_cell(s, charger)

        # Not adjacent yet, move towards it
        if self._debug:
            msg = "  → Navigating to charger\n"
            if hasattr(self, "_debug_file") and self._debug_file:
                self._debug_file.write(msg)
                self._debug_file.flush()
        return self._move_towards(s, charger, reach_adjacent=True)

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

        if self._debug:
            msg = (
                f"[Agent {s.agent_id}] _find_nearest_extractor({resource_type}): "
                f"{len(extractors)} total, {len(available)} available\n"
            )
            for e in extractors:
                msg += (
                    f"  {e.position}: uses={e.remaining_uses}, clipped={e.clipped}, cooldown={e.cooldown_remaining}\n"
                )
            if hasattr(self, "_debug_file") and self._debug_file:
                self._debug_file.write(msg)
                self._debug_file.flush()

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

        goal_cells = self._compute_goal_cells(s, target, reach_adjacent)
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
            if self._is_traversable(s, next_pos[0], next_pos[1]) or (allow_goal_block and next_pos in goal_cells):
                # Cached path is valid! Use it
                path = s.cached_path
            # If next step blocked, fall through to recompute

        # Need to recompute path
        if path is None:
            path = self._shortest_path(s, start, goal_cells, allow_goal_block)
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

        if dr == -1 and dc == 0:
            return self._actions.move.Move("north")
        if dr == 1 and dc == 0:
            return self._actions.move.Move("south")
        if dr == 0 and dc == 1:
            return self._actions.move.Move("east")
        if dr == 0 and dc == -1:
            return self._actions.move.Move("west")

        return self._actions.noop.Noop()

    def _compute_goal_cells(
        self, s: SimpleAgentState, target: tuple[int, int], reach_adjacent: bool
    ) -> list[tuple[int, int]]:
        if not reach_adjacent:
            return [target]

        goals: list[tuple[int, int]] = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = target[0] + dr, target[1] + dc
            if self._is_traversable(s, nr, nc):
                goals.append((nr, nc))

        # If no adjacent traversable tiles are known yet, allow exploring toward unknown ones
        if not goals:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = target[0] + dr, target[1] + dc
                if self._is_within_bounds(s, nr, nc) and s.occupancy[nr][nc] != CellType.OBSTACLE.value:
                    goals.append((nr, nc))
        return goals

    def _shortest_path(
        self,
        s: SimpleAgentState,
        start: tuple[int, int],
        goals: list[tuple[int, int]],
        allow_goal_block: bool,
    ) -> list[tuple[int, int]]:
        goal_set = set(goals)
        queue: deque[tuple[int, int]] = deque([start])
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

        def walkable(r: int, c: int) -> bool:
            if (r, c) in goal_set and allow_goal_block:
                return True
            return self._is_traversable(s, r, c)

        while queue:
            current = queue.popleft()
            if current in goal_set:
                return self._reconstruct_path(came_from, current)

            for nr, nc in self._neighbors(s, current):
                if (nr, nc) not in came_from and walkable(nr, nc):
                    came_from[(nr, nc)] = current
                    queue.append((nr, nc))

        return []

    def _reconstruct_path(
        self,
        came_from: dict[tuple[int, int], tuple[int, int] | None],
        current: tuple[int, int],
    ) -> list[tuple[int, int]]:
        path: list[tuple[int, int]] = []
        while came_from[current] is not None:
            path.append(current)
            prev = came_from[current]
            assert prev is not None  # Loop condition ensures this
            current = prev
        path.reverse()
        return path

    def _neighbors(self, s: SimpleAgentState, pos: tuple[int, int]) -> list[tuple[int, int]]:
        r, c = pos
        candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        return [(nr, nc) for nr, nc in candidates if self._is_within_bounds(s, nr, nc)]

    def _is_within_bounds(self, s: SimpleAgentState, r: int, c: int) -> bool:
        return 0 <= r < s.map_height and 0 <= c < s.map_width

    def _is_passable(self, s: SimpleAgentState, r: int, c: int) -> bool:
        """Check if a cell is passable."""
        if not self._is_within_bounds(s, r, c):
            return False
        return self._is_traversable(s, r, c)

    def _is_traversable(self, s: SimpleAgentState, r: int, c: int) -> bool:
        """Check if a cell is traversable (explicitly known to be free and no agent there)."""
        if not self._is_within_bounds(s, r, c):
            return False
        # Don't walk through other agents
        if (r, c) in s.agent_occupancy:
            return False
        cell = s.occupancy[r][c]
        # Only traverse cells we KNOW are free, not unknown cells
        return cell == CellType.FREE.value

    def _trace_log(self, s: SimpleAgentState) -> None:
        """Detailed trace logging."""
        extractors_known = {r: len(s.extractors[r]) for r in ["carbon", "oxygen", "germanium", "silicon"]}
        print(f"[TRACE Step {s.step_count}] Agent {s.agent_id} @ ({s.row},{s.col})")
        print(f"  Phase: {s.phase.name}, Energy: {s.energy}")
        print(f"  Inventory: C={s.carbon} O={s.oxygen} G={s.germanium} S={s.silicon} Hearts={s.hearts}")
        print(f"  Extractors known: {extractors_known}")
        # Debug: show first few extractors if any
        if s.step_count == 100 and s.heart_recipe:
            for rtype in s.heart_recipe:
                if len(s.extractors[rtype]) > 0:
                    first_3 = s.extractors[rtype][:3]
                    print(f"    {rtype}: {[(e.position, e.last_seen_step) for e in first_3]}")
        stations = f"assembler={s.stations['assembler'] is not None}"
        stations += f" chest={s.stations['chest'] is not None}"
        stations += f" charger={s.stations['charger'] is not None}"
        print(f"  Stations: {stations}")
        print(f"  Target: {s.target_position}, Target resource: {s.target_resource}")

    def _move_into_cell(self, s: SimpleAgentState, target: tuple[int, int]) -> Action:
        """Return the action that attempts to step into the target cell."""
        tr, tc = target
        if s.row == tr and s.col == tc:
            return self._actions.noop.Noop()
        dr = tr - s.row
        dc = tc - s.col
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
    def __init__(self, policy_env_info: PolicyEnvInterface, hyperparams: Optional[BaselineHyperparameters] = None):
        super().__init__(policy_env_info)
        self._agent_policies: dict[int, StatefulAgentPolicy[SimpleAgentState]] = {}
        self._hyperparams = hyperparams or BaselineHyperparameters()

    def agent_policy(self, agent_id: int) -> StatefulAgentPolicy[SimpleAgentState]:
        if agent_id not in self._agent_policies:
            self._agent_policies[agent_id] = StatefulAgentPolicy(
                BaselineAgentPolicyImpl(self._policy_env_info, agent_id, self._hyperparams),
                self._policy_env_info,
            )
        return self._agent_policies[agent_id]
