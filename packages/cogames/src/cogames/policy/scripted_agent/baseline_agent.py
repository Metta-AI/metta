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

import logging
import random
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable, Dict, Optional

from cogames.policy import AgentPolicy, StatefulPolicyImpl
from mettagrid.simulator.interface import Action, AgentObservation

if TYPE_CHECKING:
    from mettagrid.simulator import Simulation

logger = logging.getLogger(__name__)

# Debug flag - set to True to enable detailed logging
DEBUG = False

# Sentinel for agent-centric features
AGENT_SENTINEL = 0x55


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

    # Current position
    row: int = -1
    col: int = -1
    energy: int = 100

    # Inventory
    carbon: int = 0
    oxygen: int = 0
    germanium: int = 0
    silicon: int = 0
    hearts: int = 0

    # Current target
    target_position: Optional[tuple[int, int]] = None
    target_resource: Optional[str] = None

    # Map knowledge
    map_height: int = 0
    map_width: int = 0
    occupancy: list[list[int]] = None  # 1=free, 2=obstacle (initialized in reset)

    # Note: Station positions are now in shared self._stations, not per-agent

    # Track last action for position updates
    last_action: int = -1

    # Current glyph (vibe) for interacting with assembler
    current_glyph: str = "default"

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
    unreachable_extractors: dict[tuple[int, int], int] = None

    # Agent positions (cleared and refreshed every step)
    agent_occupancy: set[tuple[int, int]] = field(default_factory=set)

    # Stuck detection
    last_position: Optional[tuple[int, int]] = None
    stuck_counter: int = 0

    # Agent position update frequency (to avoid constant re-planning)
    agent_positions_last_updated: int = 0

    # Path caching for efficient navigation
    cached_path: Optional[list[tuple[int, int]]] = None
    cached_path_target: Optional[tuple[int, int]] = None
    cached_path_reach_adjacent: bool = False


class BaselineAgent:
    """
    Minimal baseline scripted agent with only essential features.

    Design principles:
    - Nearest extractor selection (distance only)
    - Simple BFS navigation
    - Basic phase transitions
    - No coordination, caching, or advanced features
    """

    def __init__(self, simulation: "Simulation"):
        """
        Args:
            simulation: The Simulation object
        """
        self._sim = simulation
        self._map_h = simulation.map_height
        self._map_w = simulation.map_width

        # Observation grid half-ranges from config
        obs_config = simulation.config.game.obs
        self._obs_hr = obs_config.height // 2  # Egocentric observation half-radius (rows)
        self._obs_wr = obs_config.width // 2  # Egocentric observation half-radius (cols)

        # Action lookup
        self._action_lookup = {name: idx for idx, name in enumerate(simulation.action_names)}
        self._action_names = simulation.action_names  # For converting indices back to names
        self._NOOP = self._action_lookup.get("noop", 0)
        self._MOVE_N = self._action_lookup.get("move_north", -1)
        self._MOVE_S = self._action_lookup.get("move_south", -1)
        self._MOVE_E = self._action_lookup.get("move_east", -1)
        self._MOVE_W = self._action_lookup.get("move_west", -1)
        self._move_deltas = {
            self._MOVE_N: (-1, 0),
            self._MOVE_S: (1, 0),
            self._MOVE_E: (0, 1),
            self._MOVE_W: (0, -1),
        }
        self._USE = self._action_lookup.get("use", -1)

        # Glyph (vibe) support for assembler - use actual VIBES order from env
        from cogames.cogs_vs_clips.vibes import VIBES

        self._glyph_name_to_id = {v.name: i for i, v in enumerate(VIBES)}
        # Vibe actions are named "change_vibe_{vibe_name}", not "change_vibe_{idx}"
        self._change_vibe_actions = {
            name: self._action_lookup.get(f"change_vibe_{name}", -1) for name in self._glyph_name_to_id.keys()
        }

        # Feature ID lookup for parsing observations
        self._fid: dict[str, int] = {f.name: f.id for f in simulation.features}
        self._to_int = int  # Helper for token parsing
        self._object_type_names = simulation.object_type_names

        # Fast lookup tables for observation feature decoding
        self._spatial_feature_key_by_id: dict[int, str] = {}
        for feature_name in ("type_id", "converting", "cooldown_remaining", "clipped", "remaining_uses"):
            fid = self._fid.get(feature_name)
            if fid is not None:
                self._spatial_feature_key_by_id[fid] = feature_name

        agent_feature_pairs = {
            "agent:group": "agent_group",
            "agent:frozen": "agent_frozen",
            "agent:orientation": "agent_orientation",
            "agent:visitation_counts": "agent_visitation_counts",
        }
        self._agent_feature_key_by_id: dict[int, str] = {}
        for feature_name, key in agent_feature_pairs.items():
            fid = self._fid.get(feature_name)
            if fid is not None:
                self._agent_feature_key_by_id[fid] = key

        # Shared knowledge across all agents
        self._extractors: dict[str, list[ExtractorInfo]] = {
            "carbon": [],
            "oxygen": [],
            "germanium": [],
            "silicon": [],
        }

        # Shared station positions (discovered by any agent)
        self._stations: dict[str, tuple[int, int] | None] = {
            "assembler": None,
            "chest": None,
            "charger": None,
        }

        # Agent states
        self._agent_states: dict[int, SimpleAgentState] = {}

        # Resource requirements for one heart
        self._heart_recipe = {"carbon": 20, "oxygen": 20, "germanium": 5, "silicon": 50}

        # Cached grid_objects for position lookups - shared across all agents per step
        self._cached_grid_objects: Optional[dict] = None
        self._cached_grid_objects_step: int = -1

    def parse_observation(
        self, obs: AgentObservation, agent_row: int, agent_col: int, debug: bool = False
    ) -> ParsedObservation:
        """Parse token-based observation into structured format.

        New format: AgentObservation with tokens (ObservationToken list)
        - Inventory is obtained via agent.inventory (not parsed here)
        - Only spatial features are parsed from observations

        Converts egocentric spatial coordinates to world coordinates using agent position.
        Agent position (agent_row, agent_col) comes from simulation.grid_objects().
        """
        # Initialize spatial data only - inventory comes from agent.inventory
        nearby_objects: dict[tuple[int, int], ObjectState] = {}

        # First pass: collect all spatial features by position
        position_features: dict[tuple[int, int], dict[str, int]] = {}

        # Parse tokens - only spatial features
        for tok in obs.tokens:
            # Get location (row, col) - already unpacked
            obs_r, obs_c = tok.location
            feature_id = tok.feature.id
            value = tok.value

            # Skip center location - that's inventory/global obs, obtained via agent.inventory
            if obs_r == self._obs_hr and obs_c == self._obs_wr:
                continue

            # Spatial features (relative to agent)
            if agent_row >= 0 and agent_col >= 0:
                # Convert observation-relative coords to world coords
                r, c = obs_r - self._obs_hr + agent_row, obs_c - self._obs_wr + agent_col
                if 0 <= r < self._map_h and 0 <= c < self._map_w:
                    pos = (r, c)
                    if pos not in position_features:
                        position_features[pos] = {}

                    # Collect all features for this position
                    feature_key = self._spatial_feature_key_by_id.get(feature_id)
                    if feature_key is not None:
                        # For type_id, only take the FIRST non-zero value (don't overwrite)
                        if feature_key == "type_id":
                            if "type_id" not in position_features[pos] and value != 0:
                                position_features[pos][feature_key] = value
                        else:
                            position_features[pos][feature_key] = value
                        continue

                    agent_feature_key = self._agent_feature_key_by_id.get(feature_id)
                    if agent_feature_key is not None:
                        position_features[pos][agent_feature_key] = value

        # Second pass: create ObjectState for each position
        for pos, features in position_features.items():
            type_id = features.get("type_id", 0)

            # Get object name
            obj_name = ""
            if 0 <= type_id < len(self._object_type_names):
                obj_name = self._object_type_names[type_id]

            # Add ALL observed cells to nearby_objects
            # Empty string means floor (passable), other names are actual objects
            nearby_objects[pos] = ObjectState(
                name=obj_name if obj_name else "floor",  # Use "floor" for empty strings
                converting=features.get("converting", 0),
                cooldown_remaining=features.get("cooldown_remaining", 0),
                clipped=features.get("clipped", 0),
                remaining_uses=features.get("remaining_uses", 999),
                # Agent features
                agent_id=features.get("agent_id", -1),
                agent_group=features.get("agent_group", -1),
                agent_frozen=features.get("agent_frozen", 0),
                agent_orientation=features.get("agent_orientation", 0),
                agent_visitation_counts=features.get("agent_visitation_counts", 0),
            )

        return ParsedObservation(
            row=agent_row,  # Position from tracking
            col=agent_col,
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
            nearby_objects=nearby_objects,  # Spatial data from observations
        )

    def reset(self, num_agents: int) -> None:
        """Reset all agent states."""
        self._agent_states.clear()
        for agent_id in range(num_agents):
            state = SimpleAgentState(
                agent_id=agent_id,
                map_height=self._map_h,
                map_width=self._map_w,
                occupancy=[[CellType.FREE.value] * self._map_w for _ in range(self._map_h)],
            )
            # Initialize mutable defaults that can't be in dataclass
            state.unreachable_extractors = {}
            self._agent_states[agent_id] = state
        # Clear extractor memory
        for resource in self._extractors:
            self._extractors[resource].clear()

    def step(self, agent_id: int, obs: AgentObservation) -> int:
        """Compute action for one agent (returns action index)."""
        s = self._agent_states[agent_id]
        s.step_count += 1

        s.agent_occupancy.clear()

        # Update state from observation
        self._update_state_from_obs(s, obs)

        # Check phase transitions
        self._update_phase(s)

        # Execute current phase
        action = self._execute_phase(s)

        # Save action for next step's position update
        s.last_action = action

        return action

    def _update_agent_position(self, s: SimpleAgentState) -> None:
        """Get agent position from simulation grid_objects().

        Optimization: Cache grid_objects() result and share across all agents in same step.
        With ~370 objects in a typical map, this reduces lookups from 1,480/step (4 agents)
        to just 370/step - a 4x speedup for position updates.
        """
        try:
            # Use cached grid_objects if it's from the current step
            if self._cached_grid_objects_step != s.step_count:
                self._cached_grid_objects = self._sim.grid_objects()
                self._cached_grid_objects_step = s.step_count

            # Find this agent in the cached grid objects
            for _id, obj in self._cached_grid_objects.items():
                if obj.get("agent_id") == s.agent_id:
                    new_row, new_col = obj.get("r", -1), obj.get("c", -1)
                    s.row, s.col = new_row, new_col
                    break
        except Exception:
            pass  # Silently fail if we can't get position

    def _update_state_from_obs(self, s: SimpleAgentState, obs: AgentObservation) -> None:
        """Update agent state from observation."""

        # STEP 1: Get agent position directly from environment
        self._update_agent_position(s)

        # STEP 2: Get inventory from observation tokens AT THE CENTER (agent's own position)
        # NOTE: Observations contain inventory for ALL visible agents. We must only read
        # from the center of the egocentric view where the observing agent is located.
        inv = {}
        features_list = self._sim.features
        center_r, center_c = self._obs_hr, self._obs_wr  # Center of egocentric observation

        for tok in obs.tokens:
            # Only read inventory from the center cell (where this agent is)
            if tok.location == (center_r, center_c):
                feature_spec = features_list[tok.feature.id]
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
        parsed = self.parse_observation(obs, s.row, s.col, debug=debug)

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

        # First pass: for everything we can see, reset to FREE (will be remarked below)
        # This clears old agent positions
        for pos in parsed.nearby_objects.keys():
            r, c = pos
            s.occupancy[r][c] = CellType.FREE.value

        # Second pass: mark obstacles
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

            # Discover stations (mark as obstacles - can't walk through them)
            if self._is_station(obj_name, "assembler"):
                s.occupancy[r][c] = CellType.OBSTACLE.value
                self._discover_station(s, pos, "assembler")
            elif self._is_station(obj_name, "chest"):
                s.occupancy[r][c] = CellType.OBSTACLE.value
                self._discover_station(s, pos, "chest")
            elif self._is_station(obj_name, "charger"):
                s.occupancy[r][c] = CellType.OBSTACLE.value
                self._discover_station(s, pos, "charger")
            elif "extractor" in obj_name:
                # Extractors are obstacles (can't walk through them)
                s.occupancy[r][c] = CellType.OBSTACLE.value
                resource_type = obj_name.replace("_extractor", "")
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
        if self._stations.get(station_key) is None:
            self._stations[station_key] = pos

    def _discover_extractor(
        self,
        s: SimpleAgentState,
        pos: tuple[int, int],
        resource_type: str,
        obj_state: ObjectState,
    ) -> None:
        extractor = None
        for existing in self._extractors[resource_type]:
            if existing.position == pos:
                extractor = existing
                break

        if extractor is None:
            extractor = ExtractorInfo(
                position=pos,
                resource_type=resource_type,
                last_seen_step=s.step_count,
            )
            self._extractors[resource_type].append(extractor)

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
        # Enter RECHARGE if energy drops below 30
        if s.energy < 30:
            if s.phase != Phase.RECHARGE:
                s.phase = Phase.RECHARGE
                # Clear extractor waiting state when leaving GATHER
                s.pending_use_resource = None
                s.pending_use_amount = 0
                s.waiting_at_extractor = None
            return

        # Stay in RECHARGE until energy is fully restored (>= 90)
        if s.phase == Phase.RECHARGE:
            if s.energy >= 90:
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
        can_assemble = (
            s.carbon >= self._heart_recipe["carbon"]
            and s.oxygen >= self._heart_recipe["oxygen"]
            and s.germanium >= self._heart_recipe["germanium"]
            and s.silicon >= self._heart_recipe["silicon"]
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

    def _calculate_deficits(self, s: SimpleAgentState) -> dict[str, int]:
        """Calculate how many more of each resource we need for a heart."""
        return {
            "carbon": max(0, self._heart_recipe["carbon"] - s.carbon),
            "oxygen": max(0, self._heart_recipe["oxygen"] - s.oxygen),
            "germanium": max(0, self._heart_recipe["germanium"] - s.germanium),
            "silicon": max(0, self._heart_recipe["silicon"] - s.silicon),
        }

    def _execute_phase(self, s: SimpleAgentState) -> int:
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
        return self._NOOP

    def _explore_frontier(self, s: SimpleAgentState) -> int:
        """
        Frontier-based exploration: move toward areas we haven't visited recently.
        Uses a 'visited' tracking system to encourage exploring new areas.
        More efficient than random walk for systematic map coverage.

        Includes occasional random moves to break out of repetitive patterns.
        """
        if s.row < 0:
            return self._NOOP

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
                (self._MOVE_N, (-1, 0)),
                (self._MOVE_S, (1, 0)),
                (self._MOVE_E, (0, 1)),
                (self._MOVE_W, (0, -1)),
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
            # Keep target if: we set it recently (within 10 steps) AND haven't reached it yet
            if s.step_count - s.exploration_target_step < 10 and (s.row, s.col) != s.exploration_target:
                # Check if target is still valid (FREE and in bounds)
                if self._is_within_bounds(s, tr, tc) and s.occupancy[tr][tc] == CellType.FREE.value:
                    return self._move_towards(s, s.exploration_target)
            # Target reached or expired, clear it
            s.exploration_target = None

        # Find nearest least-recently-visited cell
        best_target = None
        best_score = -float("inf")

        # Sample cells in expanding radius
        for radius in range(5, min(15, s.map_height // 2, s.map_width // 2)):
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

                    # Prefer unvisited or long-unvisited cells closer to us
                    score = visit_score * 10 - distance

                    if score > best_score:
                        best_score = score
                        best_target = (r, c)

            if best_target and best_score > 100:  # Found a good target
                break

        if best_target:
            # Commit to this target for multiple steps
            s.exploration_target = best_target
            s.exploration_target_step = s.step_count
            return self._move_towards(s, best_target)

        # No good target found, pick a random direction
        return random.choice([self._MOVE_N, self._MOVE_S, self._MOVE_E, self._MOVE_W])

    def _explore_directed(self, s: SimpleAgentState, target_area: tuple[int, int], radius: int = 5) -> int:
        """
        Directed exploration: move toward a specific area to explore it.
        Useful for searching specific regions of the map.
        """
        # Move toward target area
        action = self._move_towards(s, target_area)
        if action != self._NOOP:
            return action

        # If we've reached the area, use frontier exploration
        return self._explore_frontier(s)

    def _explore_until(
        self, s: SimpleAgentState, condition: Callable[[], bool], reason: str = "Exploring"
    ) -> int | None:
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

    def _explore(self, s: SimpleAgentState) -> int:
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

    def _do_gather(self, s: SimpleAgentState) -> int:
        """
        Gather resources from nearest extractors.
        Opportunistically uses ANY extractor for ANY needed resource.
        """
        # If we're waiting for an activated extractor to finish converting, just wait
        if s.pending_use_resource is not None:
            # Safety check: if we've been waiting too long, clear and retry
            s.wait_steps += 1
            if s.wait_steps > 50:  # If stuck waiting for 50 steps
                s.pending_use_resource = None
                s.pending_use_amount = 0
                s.waiting_at_extractor = None
                s.wait_steps = 0
                return self._NOOP

            return self._NOOP

        deficits = self._calculate_deficits(s)

        # If no deficits, we're done
        if all(d <= 0 for d in deficits.values()):
            s.waiting_at_extractor = None
            s.wait_steps = 0
            return self._NOOP

        # Explore until we find ANY extractor for ANY needed resource
        explore_action = self._explore_until(
            s,
            condition=lambda: self._find_any_needed_extractor(s) is not None,
            reason=f"Need extractors for: {', '.join(k for k, v in deficits.items() if v > 0)}",
        )
        if explore_action is not None:
            return explore_action

        # Found an extractor, get it
        result = self._find_any_needed_extractor(s)
        if result is None:
            # This shouldn't happen since condition just passed, but handle it
            return self._explore(s)

        extractor, resource_type = result

        # Clear exploration target - we're now targeting an extractor
        s.exploration_target = None

        s.target_resource = resource_type

        er, ec = extractor.position
        dr = abs(s.row - er)
        dc = abs(s.col - ec)
        is_adjacent = (dr == 1 and dc == 0) or (dr == 0 and dc == 1)

        # If not adjacent, move towards the extractor
        if not is_adjacent:
            s.waiting_at_extractor = None
            s.wait_steps = 0
            action = self._move_towards(s, extractor.position, reach_adjacent=True)
            if action == self._NOOP:
                return self._explore(s)
            return action

        # We're adjacent to the extractor!
        # Verify position from simulation
        # If in cooldown or converting, wait for it
        if extractor.cooldown_remaining > 0 or extractor.converting:
            s.waiting_at_extractor = extractor.position
            s.wait_steps += 1
            return self._NOOP

        # If out of uses or clipped, it's not usable - move on to next extractor
        if extractor.remaining_uses == 0 or extractor.clipped:
            s.waiting_at_extractor = None
            s.wait_steps = 0
            # Will find another extractor on next iteration
            return self._NOOP
        # Extractor is usable! Track inventory before use, then move into it to activate
        old_amount = getattr(s, resource_type, 0)

        # Calculate the move direction
        tr, tc = extractor.position
        dr = tr - s.row
        dc = tc - s.col

        action = self._move_into_cell(s, extractor.position)

        # Now set the waiting state AFTER we've logged everything
        s.pending_use_resource = resource_type
        s.pending_use_amount = old_amount
        s.waiting_at_extractor = extractor.position

        return action

    def _do_assemble(self, s: SimpleAgentState) -> int:
        """Assemble hearts at assembler."""
        # Explore until we find assembler
        explore_action = self._explore_until(
            s, condition=lambda: self._stations["assembler"] is not None, reason="Need assembler"
        )
        if explore_action is not None:
            return explore_action

        # First, ensure we have the correct glyph (heart) for assembling
        if s.current_glyph != "heart":
            vibe_action = self._change_vibe_actions["heart"]
            s.current_glyph = "heart"
            return vibe_action

        # Assembler is known, move adjacent to it then use it
        assembler = self._stations["assembler"]
        ar, ac = assembler
        dr = abs(s.row - ar)
        dc = abs(s.col - ac)
        is_adjacent = (dr == 1 and dc == 0) or (dr == 0 and dc == 1)

        if is_adjacent:
            # Adjacent - move into it to use it (assembler will consume resources and give heart)
            return self._move_into_cell(s, assembler)

        # Not adjacent yet, move towards it
        return self._move_towards(s, assembler, reach_adjacent=True)

    def _do_deliver(self, s: SimpleAgentState) -> int:
        """Deliver hearts to chest."""
        # Explore until we find chest
        explore_action = self._explore_until(
            s, condition=lambda: self._stations["chest"] is not None, reason="Need chest"
        )
        if explore_action is not None:
            return explore_action

        # Chest is known, move adjacent to it then use it
        chest = self._stations["chest"]
        cr, cc = chest
        dr = abs(s.row - cr)
        dc = abs(s.col - cc)
        is_adjacent = (dr == 1 and dc == 0) or (dr == 0 and dc == 1)

        if is_adjacent:
            # Adjacent - move into it to deliver
            return self._move_into_cell(s, chest)

        # Not adjacent yet, move towards it
        return self._move_towards(s, chest, reach_adjacent=True)

    def _do_recharge(self, s: SimpleAgentState) -> int:
        """Recharge at charger."""
        # Explore until we find charger
        explore_action = self._explore_until(
            s, condition=lambda: self._stations["charger"] is not None, reason="Need charger"
        )
        if explore_action is not None:
            return explore_action

        # Charger is known, move adjacent to it
        charger = self._stations["charger"]
        chr, chc = charger
        dr = abs(s.row - chr)
        dc = abs(s.col - chc)
        is_adjacent = (dr == 1 and dc == 0) or (dr == 0 and dc == 1)

        if is_adjacent:
            # Adjacent to charger - NOOP to recharge
            return self._NOOP

        # Not adjacent yet, move towards it
        return self._move_towards(s, charger, reach_adjacent=True)

    def _do_unclip(self, s: SimpleAgentState) -> int:
        """Unclip extractors (TODO: implement)."""
        # TODO: Find nearest clipped extractor, go to it, activate it to unclip
        # For now, just return to GATHER phase
        s.phase = Phase.GATHER
        return self._NOOP

    def _find_nearest_extractor(self, s: SimpleAgentState, resource_type: str) -> Optional[ExtractorInfo]:
        """Find the nearest AVAILABLE extractor of the given type."""
        extractors = self._extractors.get(resource_type, [])
        if not extractors:
            return None

        # Filter out clipped (depleted) extractors
        available = [e for e in extractors if not e.clipped]
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
    ) -> int:
        """Pathfind toward a target using BFS with obstacle awareness.

        Uses path caching to avoid recomputing BFS every step - only recomputes when:
        - Target changed
        - Path parameters changed (reach_adjacent)
        - Next step is blocked
        - Path completed or invalid
        """

        start = (s.row, s.col)
        if start == target and not reach_adjacent:
            return self._NOOP

        goal_cells = self._compute_goal_cells(s, target, reach_adjacent)
        if not goal_cells:
            return self._NOOP

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
            for dr in range(-2, 3):
                row_str = "    "
                for dc in range(-2, 3):
                    r, c = start[0] + dr, start[1] + dc
                    if self._is_within_bounds(s, r, c):
                        cell = s.occupancy[r][c]
                        if cell == CellType.FREE.value:
                            row_str += "."
                        elif cell == CellType.OBSTACLE.value:
                            row_str += "#"
                        else:
                            row_str += "?"
                    else:
                        row_str += "X"
            for dr in range(-2, 3):
                row_str = "    "
                for dc in range(-2, 3):
                    r, c = target[0] + dr, target[1] + dc
                    if self._is_within_bounds(s, r, c):
                        cell = s.occupancy[r][c]
                        if cell == CellType.FREE.value:
                            row_str += "."
                        elif cell == CellType.OBSTACLE.value:
                            row_str += "#"
                        else:
                            row_str += "?"
                    else:
                        row_str += "X"
            return self._NOOP

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
            return self._MOVE_N
        if dr == 1 and dc == 0:
            return self._MOVE_S
        if dr == 0 and dc == 1:
            return self._MOVE_E
        if dr == 0 and dc == -1:
            return self._MOVE_W

        return self._NOOP

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
            current = came_from[current]
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
        extractors_known = {r: len(self._extractors[r]) for r in self._heart_recipe}
        print(f"[TRACE Step {s.step_count}] Agent {s.agent_id} @ ({s.row},{s.col})")
        print(f"  Phase: {s.phase.name}, Energy: {s.energy}")
        print(f"  Inventory: C={s.carbon} O={s.oxygen} G={s.germanium} S={s.silicon} Hearts={s.hearts}")
        print(f"  Extractors known: {extractors_known}")
        # Debug: show first few extractors if any
        if s.step_count == 100:
            for rtype in self._heart_recipe:
                if len(self._extractors[rtype]) > 0:
                    first_3 = self._extractors[rtype][:3]
                    print(f"    {rtype}: {[(e.position, e.last_seen_step) for e in first_3]}")
        stations = f"assembler={self._stations['assembler'] is not None}"
        stations += f" chest={self._stations['chest'] is not None}"
        stations += f" charger={self._stations['charger'] is not None}"
        print(f"  Stations: {stations}")
        print(f"  Target: {s.target_position}, Target resource: {s.target_resource}")

    def _move_into_cell(self, s: SimpleAgentState, target: tuple[int, int]) -> int:
        """Return the action that attempts to step into the target cell."""
        tr, tc = target
        if s.row == tr and s.col == tc:
            return self._NOOP
        dr = tr - s.row
        dc = tc - s.col
        if dr == -1 and self._MOVE_N != -1:
            return self._MOVE_N
        if dr == 1 and self._MOVE_S != -1:
            return self._MOVE_S
        if dc == 1 and self._MOVE_E != -1:
            return self._MOVE_E
        if dc == -1 and self._MOVE_W != -1:
            return self._MOVE_W
        # Fallback to pathfinder if offsets unexpected
        return self._move_towards(s, target, allow_goal_block=True)


# ============================================================================
# Policy Wrapper Classes
# ============================================================================


class BaselinePolicyImpl(StatefulPolicyImpl[SimpleAgentState]):
    """Implementation that wraps BaselineAgent."""

    def __init__(self, simulation: "Simulation"):
        self._agent = BaselineAgent(simulation)
        self._sim = simulation

    def agent_state(self, agent_id: int = 0) -> SimpleAgentState:
        """Get initial state for an agent."""
        # Make sure agent states are initialized
        if agent_id not in self._agent._agent_states:
            self._agent._agent_states[agent_id] = SimpleAgentState(
                agent_id=agent_id,
                map_height=self._agent._map_h,
                map_width=self._agent._map_w,
                occupancy=[[CellType.FREE.value] * self._agent._map_w for _ in range(self._agent._map_h)],
            )
        return self._agent._agent_states[agent_id]

    def step_with_state(self, obs: AgentObservation, state: SimpleAgentState) -> tuple[Action, SimpleAgentState]:
        """Compute action and return updated state."""
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


class BaselinePolicy:
    """Policy class for simple baseline agent.

    This policy requires a Simulation object for accessing grid_objects()
    to get absolute agent positions. Pass it via reset(simulation=sim).
    """

    def __init__(self):
        """Initialize policy (simulation will be provided via reset)."""
        self._sim = None
        self._impl = None
        self._agent_policies: Dict[int, AgentPolicy] = {}

    def reset(self, simulation: "Simulation" = None) -> None:
        """Reset all agent states.

        Args:
            simulation: The Simulation object (needed for grid_objects access)
        """
        if simulation is None:
            raise RuntimeError("BaselinePolicy requires simulation parameter in reset()")

        self._sim = simulation
        self._impl = BaselinePolicyImpl(simulation)
        self._agent_policies.clear()

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        """Get an AgentPolicy instance for a specific agent."""
        if self._impl is None:
            raise RuntimeError("Policy not initialized - call reset(simulation=sim) first")

        # Create agent policies lazily
        if agent_id not in self._agent_policies:
            from cogames.policy import StatefulAgentPolicy

            self._agent_policies[agent_id] = StatefulAgentPolicy(self._impl, agent_id)
        return self._agent_policies[agent_id]
