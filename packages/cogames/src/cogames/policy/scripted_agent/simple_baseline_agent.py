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
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from cogames.cogs_vs_clips.env import MettaGridEnv
    from cogames.cogs_vs_clips.observation import MettaGridObservation

logger = logging.getLogger(__name__)
# Observation grid half-ranges (hardcoded for now)
OBS_HR = 7  # rows
OBS_WR = 7  # cols

# Sentinel for agent-centric features
AGENT_SENTINEL = 0x55


class CellType(Enum):
    """Occupancy map cell states."""

    UNKNOWN = 0  # Not yet observed
    FREE = 1  # Passable (can walk through)
    OBSTACLE = 2  # Impassable (walls, stations, extractors)


class Phase(Enum):
    """Simple phase states for the baseline agent."""

    EXPLORE = "explore"  # Finding extractors and stations
    GATHER = "gather"  # Collecting resources from extractors
    DEPOSIT = "deposit"  # Taking resources to assembler
    ASSEMBLE = "assemble"  # Assembling hearts
    DELIVER = "deliver"  # Delivering hearts to chest
    RECHARGE = "recharge"  # Recharging at charger


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

    # Nearby objects with full state (position -> ObjectState)
    nearby_objects: dict[tuple[int, int], ObjectState]


@dataclass
class SimpleAgentState:
    """State for a single agent."""

    agent_id: int
    phase: Phase = Phase.EXPLORE
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

    # Current target
    target_position: Optional[tuple[int, int]] = None
    target_resource: Optional[str] = None

    # Map knowledge
    map_height: int = 0
    map_width: int = 0
    occupancy: list[list[int]] = None  # 0=unknown, 1=free, 2=wall

    # Note: Station positions are now in shared self._stations, not per-agent

    # Exploration
    explore_step: int = 0
    max_explore_steps: int = 100

    # Track last action for position updates
    last_action: int = -1

    # Extractor waiting
    waiting_at_extractor: Optional[tuple[int, int]] = None
    wait_steps: int = 0


class SimpleBaselineAgent:
    """
    Minimal baseline scripted agent with only essential features.

    Design principles:
    - Nearest extractor selection (distance only)
    - Simple BFS navigation
    - Basic phase transitions
    - No coordination, caching, or advanced features
    """

    def __init__(self, env: MettaGridEnv):
        self._env = env
        self._map_h = env.map_height
        self._map_w = env.map_width

        # Action lookup
        self._action_lookup = {name: idx for idx, name in enumerate(env.action_names)}
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

        # Feature ID lookup for parsing observations
        feats = env.observation_features
        self._fid: dict[str, int] = {f.name: f.id for f in feats.values()}
        self._to_int = int  # Helper for token parsing

        # Object type names
        self._object_type_names: list[str] = env.object_type_names

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
        self._heart_recipe = {"carbon": 1, "oxygen": 1, "germanium": 1, "silicon": 1}

        logger.info("[SimpleBaseline] Initialized for map %dx%d", self._map_h, self._map_w)
        print(f"[SimpleBaseline] Initialized for map {self._map_h}x{self._map_w}")

    def parse_observation(
        self, obs: MettaGridObservation, agent_row: int, agent_col: int, debug: bool = False
    ) -> ParsedObservation:
        """Parse token-based observation into structured format.

        Token format: [packed_pos, feature_id, value]
        - packed_pos = (obs_r << 4) | obs_c (egocentric, relative to agent)
        - Special: packed_pos = 0x55 (85) is sentinel for agent-centric features (inventory, etc.)
        - feature_id identifies what the value represents
        - value is the actual data

        Converts egocentric spatial coordinates to world coordinates using agent position.
        Agent position (agent_row, agent_col) comes from env.c_env.grid_objects().
        """
        ti = self._to_int

        # Initialize parsed data - position comes from tracking, not observations
        row = agent_row
        col = agent_col
        energy = 0
        carbon = 0
        oxygen = 0
        germanium = 0
        silicon = 0
        nearby_objects: dict[tuple[int, int], ObjectState] = {}

        # Feature IDs - spatial object features
        # First pass: collect all features by position
        position_features: dict[tuple[int, int], dict[str, int]] = {}

        # Parse tokens
        for tok in obs:
            packed = ti(tok[0])
            feature_id = ti(tok[1])
            value = ti(tok[2])

            # Agent-centric features (use sentinel packed pos)
            if packed == AGENT_SENTINEL:
                if feature_id == self._fid.get("inv:energy"):
                    energy = value
                elif feature_id == self._fid.get("inv:carbon"):
                    carbon = value
                elif feature_id == self._fid.get("inv:oxygen"):
                    oxygen = value
                elif feature_id == self._fid.get("inv:germanium"):
                    germanium = value
                elif feature_id == self._fid.get("inv:silicon"):
                    silicon = value
            # Spatial features (relative to agent)
            elif agent_row >= 0 and agent_col >= 0:
                # Convert observation-relative coords to world coords
                obs_r, obs_c = packed >> 4, packed & 0x0F
                r, c = obs_r - OBS_HR + agent_row, obs_c - OBS_WR + agent_col
                if 0 <= r < self._map_h and 0 <= c < self._map_w:
                    pos = (r, c)
                    if pos not in position_features:
                        position_features[pos] = {}

                    # Collect all features for this position
                    if feature_id == self._fid.get("type_id"):
                        position_features[pos]["type_id"] = value
                    elif feature_id == self._fid.get("converting"):
                        position_features[pos]["converting"] = value
                    elif feature_id == self._fid.get("cooldown_remaining"):
                        position_features[pos]["cooldown_remaining"] = value
                    elif feature_id == self._fid.get("clipped"):
                        position_features[pos]["clipped"] = value
                    elif feature_id == self._fid.get("remaining_uses"):
                        position_features[pos]["remaining_uses"] = value
                    # Agent-specific features
                    elif feature_id == self._fid.get("agent:group"):
                        position_features[pos]["agent_group"] = value
                    elif feature_id == self._fid.get("agent:frozen"):
                        position_features[pos]["agent_frozen"] = value
                    elif feature_id == self._fid.get("agent:orientation"):
                        position_features[pos]["agent_orientation"] = value
                    elif feature_id == self._fid.get("agent:visitation_counts"):
                        position_features[pos]["agent_visitation_counts"] = value

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
            row=row,
            col=col,
            energy=energy,
            carbon=carbon,
            oxygen=oxygen,
            germanium=germanium,
            silicon=silicon,
            nearby_objects=nearby_objects,
        )

    def reset(self, num_agents: int) -> None:
        """Reset all agent states."""
        self._agent_states.clear()
        for agent_id in range(num_agents):
            self._agent_states[agent_id] = SimpleAgentState(
                agent_id=agent_id,
                map_height=self._map_h,
                map_width=self._map_w,
                occupancy=[[CellType.UNKNOWN.value] * self._map_w for _ in range(self._map_h)],
                max_explore_steps=80,  # Stay local to base
            )
        # Clear extractor memory
        for resource in self._extractors:
            self._extractors[resource].clear()

    def step(self, agent_id: int, obs: MettaGridObservation) -> int:
        """Compute action for one agent."""
        s = self._agent_states[agent_id]
        s.step_count += 1

        # Update state from observation
        self._update_state_from_obs(s, obs)

        # Trace logging
        if s.step_count % 20 == 0:
            self._trace_log(s)

        # Check phase transitions
        self._update_phase(s)

        # Execute current phase
        action = self._execute_phase(s)

        # Save action for next step's position update
        s.last_action = action

        # Debug: Log action for first few steps and when position doesn't change
        if s.step_count <= 10 or s.step_count % 20 == 0:
            action_name = "?"
            for name, idx in self._action_lookup.items():
                if idx == action:
                    action_name = name
                    break
            print(f"[Agent {s.agent_id} Step {s.step_count}] Returning action: {action} ({action_name})")

        return action

    def _update_agent_position(self, s: SimpleAgentState) -> None:
        """Get agent position directly from environment."""
        try:
            prev_pos = (s.row, s.col)
            for _id, obj in self._env.c_env.grid_objects().items():
                if obj.get("agent_id") == s.agent_id:
                    new_row, new_col = obj.get("r", -1), obj.get("c", -1)
                    if s.row == -1:  # First time
                        print(f"[Agent {s.agent_id}] Initial position from env: ({new_row}, {new_col})")
                    elif (new_row, new_col) != prev_pos and s.step_count % 20 == 0:
                        print(f"[Agent {s.agent_id}] Env position: {prev_pos} -> ({new_row}, {new_col})")
                    s.row, s.col = new_row, new_col
                    break
        except Exception as e:
            print(f"[Agent {s.agent_id}] Warning: could not get position from env: {e}")

        # Mark failed movement attempts as obstacles
        if s.last_action in self._move_deltas:
            dr, dc = self._move_deltas[s.last_action]
            target = (prev_pos[0] + dr, prev_pos[1] + dc)
            if (s.row, s.col) == prev_pos and self._is_within_bounds(s, target[0], target[1]):
                if s.occupancy[target[0]][target[1]] != 2:
                    s.occupancy[target[0]][target[1]] = 2

    def _update_state_from_obs(self, s: SimpleAgentState, obs: MettaGridObservation) -> None:
        """Update agent state from observation."""

        # STEP 1: Get agent position directly from environment
        self._update_agent_position(s)

        # STEP 2: Parse observation with known position
        debug = s.step_count == 1
        parsed = self.parse_observation(obs, s.row, s.col, debug=debug)

        # STEP 3: Update agent state from parsed obs
        s.energy = parsed.energy
        s.carbon = parsed.carbon
        s.oxygen = parsed.oxygen
        s.germanium = parsed.germanium
        s.silicon = parsed.silicon

        # STEP 4: Discover objects (updates occupancy map)
        self._discover_objects(s, parsed)

    def _discover_objects(self, s: SimpleAgentState, parsed: ParsedObservation) -> None:
        """Discover extractors and stations from parsed observation."""
        if s.row < 0:
            return

        # Debug: log number of observed cells
        if s.step_count == 20:
            print(f"[Agent {s.agent_id}] Observed {len(parsed.nearby_objects)} cells at step {s.step_count}")
            print(f"   Agent at ({s.row}, {s.col})")
            print(f"   Sample positions: {list(parsed.nearby_objects.keys())[:10]}")

        # Process each observed cell and mark in occupancy map
        for pos, obj_state in parsed.nearby_objects.items():
            r, c = pos
            obj_name = obj_state.name.lower()

            # Check if this is a wall or impassable obstacle
            is_wall = "wall" in obj_name or "#" in obj_name or obj_name in ["wall", "obstacle"]

            if is_wall:
                # Mark walls as obstacles (impassable)
                s.occupancy[r][c] = CellType.OBSTACLE.value
                if s.step_count < 50:  # Debug: log first few walls discovered
                    print(f"[Agent {s.agent_id}] Discovered wall at {pos}")
                continue

            # Check if this is empty floor (passable)
            is_floor = obj_name in ["floor", "", "empty", "agent"]

            if is_floor:
                # Mark floor as passable (only if not already marked as obstacle)
                if s.occupancy[r][c] != CellType.OBSTACLE.value:
                    s.occupancy[r][c] = CellType.FREE.value
                continue

            # Default: treat any non-agent object as an obstacle
            s.occupancy[r][c] = CellType.OBSTACLE.value

            # Stations (shared across all agents)
            if "assembler" in obj_name:
                if self._stations["assembler"] is None:
                    self._stations["assembler"] = pos
                    print(f"[Agent {s.agent_id}] Discovered assembler at {pos}")

            elif "chest" in obj_name:
                if self._stations["chest"] is None:
                    self._stations["chest"] = pos
                    print(f"[Agent {s.agent_id}] Discovered chest at {pos}")

            elif "charger" in obj_name:
                if self._stations["charger"] is None:
                    self._stations["charger"] = pos
                    print(f"[Agent {s.agent_id}] Discovered charger at {pos}")

            # Extractors
            elif "extractor" in obj_name:
                resource_type = obj_name.lower().replace("_extractor", "")

                if resource_type:
                    # Find existing extractor or create new one
                    extractor = None
                    for e in self._extractors[resource_type]:
                        if e.position == pos:
                            extractor = e
                            break

                    if extractor is None:
                        # New extractor discovered
                        extractor = ExtractorInfo(
                            position=pos,
                            resource_type=resource_type,
                            last_seen_step=s.step_count,
                        )
                        self._extractors[resource_type].append(extractor)
                        print(f"[Agent {s.agent_id}] Discovered {resource_type} extractor at {pos}")

                    # Update extractor state from observation
                    extractor.last_seen_step = s.step_count
                    extractor.converting = obj_state.converting > 0
                    extractor.cooldown_remaining = obj_state.cooldown_remaining
                    extractor.clipped = obj_state.clipped > 0
                    extractor.remaining_uses = obj_state.remaining_uses

                    # Debug: show state for first few observations
                    if s.step_count < 100 and extractor.converting:
                        state_str = f"converting={extractor.converting}"
                        state_str += f", cooldown={extractor.cooldown_remaining}, clipped={extractor.clipped}"
                        print(f"  -> {resource_type} @ {pos}: {state_str}")

    def _update_phase(self, s: SimpleAgentState) -> None:
        """Update agent phase based on current state."""
        # Recharge if low energy
        if s.energy < 30 and self._stations["charger"] is not None:
            if s.phase != Phase.RECHARGE:
                print(f"[Agent {s.agent_id}] Phase: {s.phase.name} -> RECHARGE (energy={s.energy})")
                s.phase = Phase.RECHARGE
                s.target_position = self._stations["charger"]
            return

        # Stop recharging when full
        if s.phase == Phase.RECHARGE and s.energy >= 90:
            print(f"[Agent {s.agent_id}] Phase: RECHARGE -> EXPLORE (energy={s.energy})")
            s.phase = Phase.EXPLORE
            s.target_position = None
            return

        # Deliver hearts if we have any
        # TODO: Need to check actual heart inventory from obs

        # Assemble if we have all resources and know where assembler is
        if self._stations["assembler"] is not None:
            can_assemble = (
                s.carbon >= self._heart_recipe["carbon"]
                and s.oxygen >= self._heart_recipe["oxygen"]
                and s.germanium >= self._heart_recipe["germanium"]
                and s.silicon >= self._heart_recipe["silicon"]
            )
            if can_assemble and s.phase != Phase.ASSEMBLE:
                print(f"[Agent {s.agent_id}] Phase: {s.phase.name} -> ASSEMBLE (resources ready)")
                s.phase = Phase.ASSEMBLE
                s.target_position = self._stations["assembler"]
                return

        # Deposit if inventory has something but can't assemble yet and we know where assembler is
        total_resources = s.carbon + s.oxygen + s.germanium + s.silicon
        if total_resources > 0 and self._stations["assembler"] is not None and s.phase == Phase.GATHER:
            # Check if we need more resources
            deficits = self._calculate_deficits(s)
            if all(deficit <= 0 for deficit in deficits.values()):
                # Have everything, go assemble
                print(f"[Agent {s.agent_id}] Phase: GATHER -> ASSEMBLE (all resources collected)")
                s.phase = Phase.ASSEMBLE
                s.target_position = self._stations["assembler"]
                return

        # Gather if we know extractors and need resources
        if s.phase == Phase.EXPLORE:
            # Check if we've explored enough or found key stations
            has_stations = self._stations["assembler"] is not None and self._stations["chest"] is not None
            # For minimal baseline: transition to GATHER if we have stations and SOME extractors
            # The gather phase will continue exploring if needed extractors aren't found
            num_extractors_found = sum(len(self._extractors[r]) for r in self._heart_recipe)
            has_some_extractors = num_extractors_found >= 2  # At least 2 different extractor types

            if has_stations and has_some_extractors:
                found_types = [r for r in self._heart_recipe if len(self._extractors[r]) > 0]
                print(
                    f"[Agent {s.agent_id}] Phase: EXPLORE -> GATHER "
                    f"(found stations + {len(found_types)}/4 extractors: {found_types})"
                )
                s.phase = Phase.GATHER
                return

            # Continue exploring until we find stations and some resources
            # Will find remaining extractors during gather phase if needed

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
        if s.phase == Phase.EXPLORE:
            return self._do_explore(s)
        elif s.phase == Phase.GATHER:
            return self._do_gather(s)
        elif s.phase == Phase.DEPOSIT:
            return self._do_deposit(s)
        elif s.phase == Phase.ASSEMBLE:
            return self._do_assemble(s)
        elif s.phase == Phase.DELIVER:
            return self._do_deliver(s)
        elif s.phase == Phase.RECHARGE:
            return self._do_recharge(s)
        return self._NOOP

    def _do_explore(self, s: SimpleAgentState) -> int:
        """Simple exploration: random walk with momentum."""
        s.explore_step += 1

        if s.row < 0:
            return self._NOOP

        # Pick a random direction every 5-10 steps and move in that direction
        # This creates a random walk that covers ground
        import random

        if not hasattr(s, "explore_direction") or s.explore_step % 7 == 0:
            # Pick a new random direction
            s.explore_direction = random.choice([self._MOVE_N, self._MOVE_S, self._MOVE_E, self._MOVE_W])

        return s.explore_direction

    def _do_gather(self, s: SimpleAgentState) -> int:
        """Gather resources from nearest extractors."""
        # Find which resource we need most
        deficits = self._calculate_deficits(s)
        needed_resource = max(deficits.items(), key=lambda x: x[1])

        if needed_resource[1] <= 0:
            # Don't need any more resources, clear waiting state
            s.waiting_at_extractor = None
            s.wait_steps = 0
            if s.step_count % 50 == 0:
                inv = f"C={s.carbon} O={s.oxygen} G={s.germanium} S={s.silicon}"
                print(f"[Agent {s.agent_id}] All resources collected! Inv: {inv}")
            return self._NOOP

        resource_type = needed_resource[0]

        # If we're waiting but now targeting a different resource, clear waiting
        # (This means we successfully collected the previous resource)
        if s.waiting_at_extractor is not None and s.target_resource != resource_type:
            print(f"[Agent {s.agent_id}] Collected resource! Moving to {resource_type}")
            s.waiting_at_extractor = None
            s.wait_steps = 0

        s.target_resource = resource_type

        # Debug logging
        if s.step_count % 50 == 0:
            print(f"[Agent {s.agent_id}] Gathering {resource_type}, deficits: {deficits}")

        # Find nearest extractor of that type
        extractor = self._find_nearest_extractor(s, resource_type)
        if extractor is None:
            # Don't know any extractors of this type, explore more
            if s.step_count % 50 == 0:
                print(f"[Agent {s.agent_id}] No {resource_type} extractor found, exploring")
            return self._do_explore(s)

        # Debug: log navigation
        if s.step_count % 50 == 0:
            print(
                f"[Agent {s.agent_id}] At ({s.row},{s.col}), target {resource_type} extractor at {extractor.position}"
            )

        # Check if we're adjacent to the extractor
        er, ec = extractor.position
        dr = abs(s.row - er)
        dc = abs(s.col - ec)
        is_adjacent = (dr == 1 and dc == 0) or (dr == 0 and dc == 1)
        at_extractor = dr == 0 and dc == 0

        # Debug: check position relationship
        if s.step_count % 50 == 0 and (is_adjacent or at_extractor):
            pos_info = f"at ({s.row},{s.col}), extractor at ({er},{ec}), dr={dr}, dc={dc}"
            state_info = f"adjacent={is_adjacent}, at_extractor={at_extractor}"
            print(f"[Agent {s.agent_id}] Position check: {pos_info}, {state_info}")

        if is_adjacent or at_extractor:
            # DETAILED DEBUG: Log inventory and extractor state
            inv = f"C={s.carbon} O={s.oxygen} G={s.germanium} S={s.silicon}"
            ext_state = f"cd={extractor.cooldown_remaining} conv={extractor.converting} clip={extractor.clipped}"
            wait_state = f"waiting_at={s.waiting_at_extractor} wait_steps={s.wait_steps}"
            print(f"[Agent {s.agent_id} Step {s.step_count}] ADJACENT to {resource_type}:")
            print(f"  {inv}, {ext_state}, {wait_state}")

            # Adjacent to extractor - activate it by moving toward it
            if extractor.cooldown_remaining > 0 or extractor.converting:
                # Extractor is busy, wait for it
                s.waiting_at_extractor = extractor.position
                s.wait_steps += 1
                # Oxygen extractors have very long cooldowns (~90 steps), so wait longer
                max_wait = 150
                if s.wait_steps > max_wait:
                    msg = f"Timeout waiting for {resource_type} @ {extractor.position} (waited {s.wait_steps} steps)"
                    print(f"[Agent {s.agent_id}] {msg}, trying different extractor")
                    s.waiting_at_extractor = None
                    s.wait_steps = 0
                    return self._do_explore(s)
                print(f"[Agent {s.agent_id}] → Returning NOOP (extractor busy)")
                return self._NOOP
            else:
                # Extractor is ready - activate by attempting to move into it
                if s.waiting_at_extractor == extractor.position and s.wait_steps > 0:
                    # We just activated it, wait for resource to appear
                    s.wait_steps += 1
                    max_wait = 150
                    if s.wait_steps > max_wait:
                        msg = f"Timeout waiting for {resource_type} to appear (waited {s.wait_steps} steps)"
                        print(f"[Agent {s.agent_id}] {msg}")
                        s.waiting_at_extractor = None
                        s.wait_steps = 0
                        return self._do_explore(s)
                    print(f"[Agent {s.agent_id}] → Returning NOOP (waiting for resource, step {s.wait_steps})")
                    return self._NOOP
                else:
                    # Activate the extractor by attempting to step onto it
                    s.waiting_at_extractor = extractor.position
                    s.wait_steps = 1
                    action = self._move_into_cell(s, extractor.position)
                    action_name = "?"
                    for name, idx in self._action_lookup.items():
                        if idx == action:
                            action_name = name
                            break
                    print(f"[Agent {s.agent_id}] → ACTIVATING with action: {action_name} (toward {extractor.position})")
                    # Move toward the extractor to activate it
                    return action
        else:
            # Not adjacent yet, clear waiting state and move closer
            if s.waiting_at_extractor is not None:
                print(f"[Agent {s.agent_id}] Moved away from extractor, clearing wait state")
                s.waiting_at_extractor = None
                s.wait_steps = 0

            # Move toward extractor (but not onto it - BFS will stop us adjacent)
            action = self._move_towards(s, extractor.position, reach_adjacent=True)

            # If no path exists (action is NOOP), explore to discover more of the map
            if action == self._NOOP:
                if s.step_count % 50 == 0:
                    print(f"[Agent {s.agent_id}] No path to {resource_type} @ {extractor.position}, exploring")
                return self._do_explore(s)

            if s.step_count % 50 == 0:
                action_name = "?"
                for name, idx in self._action_lookup.items():
                    if idx == action:
                        action_name = name
                        break
                print(f"[Agent {s.agent_id}] Moving toward {extractor.position}: {action_name}")
            return action

    def _do_deposit(self, s: SimpleAgentState) -> int:
        """Deposit resources at assembler."""
        assembler = self._stations["assembler"]
        if assembler is None:
            return self._NOOP

        if (s.row, s.col) == assembler:
            # At assembler, use it to deposit
            return self._USE

        return self._move_towards(s, assembler, allow_goal_block=True)

    def _do_assemble(self, s: SimpleAgentState) -> int:
        """Assemble hearts at assembler."""
        assembler = self._stations["assembler"]
        if assembler is None:
            return self._NOOP

        if (s.row, s.col) == assembler:
            # At assembler, use it to assemble
            # After using, should transition to gather or deliver phase
            return self._USE

        return self._move_towards(s, assembler, allow_goal_block=True)

    def _do_deliver(self, s: SimpleAgentState) -> int:
        """Deliver hearts to chest."""
        chest = self._stations["chest"]
        if chest is None:
            return self._NOOP

        if (s.row, s.col) == chest:
            # At chest, use it to deliver
            return self._USE

        return self._move_towards(s, chest, allow_goal_block=True)

    def _do_recharge(self, s: SimpleAgentState) -> int:
        """Recharge at charger."""
        charger = self._stations["charger"]
        if charger is None:
            return self._NOOP

        if (s.row, s.col) == charger:
            # At charger, wait to recharge
            return self._NOOP

        return self._move_towards(s, charger, allow_goal_block=True)

    def _find_nearest_extractor(self, s: SimpleAgentState, resource_type: str) -> Optional[ExtractorInfo]:
        """Find the nearest AVAILABLE extractor of the given type."""
        extractors = self._extractors.get(resource_type, [])
        if not extractors:
            return None

        # Filter out clipped (depleted) extractors
        available = [e for e in extractors if not e.clipped]
        if not available:
            if s.step_count % 100 == 0:
                print(f"[Agent {s.agent_id}] All {resource_type} extractors are clipped!")
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
        """Pathfind toward a target using BFS with obstacle awareness."""

        start = (s.row, s.col)
        if start == target and not reach_adjacent:
            return self._NOOP

        goal_cells = self._compute_goal_cells(s, target, reach_adjacent)
        if not goal_cells:
            return self._NOOP

        path = self._shortest_path(s, start, goal_cells, allow_goal_block)
        if not path:
            return self._NOOP

        # First step after start
        next_pos = path[0]
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
        if not self._is_within_bounds(s, r, c):
            return False
        cell = s.occupancy[r][c]
        if cell == CellType.OBSTACLE.value:
            return False
        return True

    def _trace_log(self, s: SimpleAgentState) -> None:
        """Detailed trace logging."""
        extractors_known = {r: len(self._extractors[r]) for r in self._heart_recipe}
        print(f"[TRACE Step {s.step_count}] Agent {s.agent_id} @ ({s.row},{s.col})")
        print(f"  Phase: {s.phase.name}, Energy: {s.energy}, Explore step: {s.explore_step}/{s.max_explore_steps}")
        print(f"  Inventory: C={s.carbon} O={s.oxygen} G={s.germanium} S={s.silicon}")
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
