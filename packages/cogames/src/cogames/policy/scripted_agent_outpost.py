"""Enhanced scripted agent for exploration experiments with tunable hyperparameters.

Key enhancements:
- Multi-extractor tracking: Discovers and tracks multiple extractors of same type
- Cooldown awareness: Estimates cooldown status and rotates between extractors
- Depletion monitoring: Tracks usage and explores for alternatives before running out
- Energy-aware pathfinding: Checks energy sufficiency before committing to distant targets
- Lévy flight exploration: Efficient foraging strategy based on animal behavior
- Tunable hyperparameters: Adjust behavior for different experiment types

Still observation-based: No global knowledge, discovers stations visually.
Based on Lévy flight foraging: https://en.wikipedia.org/wiki/Lévy_flight
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from cogames.policy.interfaces import AgentPolicy, Policy, StatefulAgentPolicy, StatefulPolicyImpl
from mettagrid import MettaGridAction, MettaGridEnv, MettaGridObservation, dtype_actions

logger = logging.getLogger("cogames.policy.scripted_agent_outpost")


# ===============================
# Enums & Data
# ===============================


@dataclass
class Hyperparameters:
    """Tunable hyperparameters for agent behavior."""

    # Exploration
    exploration_strategy: str = "frontier"  # "frontier", "levy", "mixed"
    levy_alpha: float = 1.5  # Lévy flight exponent (1-2), 1.5 is good balance
    exploration_radius: int = 40  # Max distance to explore from base

    # Energy management
    energy_buffer: int = 20  # Safety margin for energy calculations
    min_energy_for_silicon: int = 70  # Min energy before silicon harvesting
    charger_search_threshold: int = 40  # Search for charger below this

    # Resource strategy
    prefer_nearby: bool = True  # Prefer closer extractors over farther
    cooldown_tolerance: int = 20  # Max turns to wait for cooldown
    depletion_threshold: float = 0.2  # Explore when extractor at 20% uses left

    # Efficiency
    track_efficiency: bool = True  # Learn which extractors give more output
    efficiency_weight: float = 0.3  # Weight for efficiency vs distance (0-1)

    # Pathfinding
    use_astar: bool = True  # Use A* pathfinding for long distances
    astar_threshold: int = 20  # Use A* for distances >= this (BFS for shorter)

    # Cooldown waiting
    enable_cooldown_waiting: bool = True  # Wait near extractors on cooldown
    max_cooldown_wait: int = 100  # Max turns to wait for cooldown

    # Misc
    max_wait_turns: int = 50  # Max turns to wait before changing strategy


@dataclass
class ExtractorInfo:
    """Information about a discovered extractor."""

    position: Tuple[int, int]
    resource_type: str  # "carbon", "oxygen", "germanium", "silicon", "charger"
    station_name: str  # Full station name from game

    # Usage tracking
    last_used_step: int = -1  # When we last used it
    total_harvests: int = 0  # How many times we've used it
    total_output: int = 0  # Total resources gained from this extractor

    # Estimates (learned from observations)
    estimated_cooldown: int = 100  # Estimated cooldown duration
    estimated_uses_left: float = 1.0  # Fraction of uses remaining (0-1)

    def is_available(self, current_step: int) -> bool:
        """Check if extractor is off cooldown."""
        if self.last_used_step < 0:
            return True  # Never used
        return current_step >= self.last_used_step + self.estimated_cooldown

    def is_depleted(self) -> bool:
        """Check if extractor is likely depleted."""
        return self.estimated_uses_left <= 0.05

    def avg_output(self) -> float:
        """Average output per use."""
        if self.total_harvests == 0:
            return 0.0
        return self.total_output / self.total_harvests

    def update_after_use(self, output: int, current_step: int, cooldown: int = 100):
        """Update stats after using extractor."""
        self.last_used_step = current_step
        self.total_harvests += 1
        self.total_output += output
        self.estimated_cooldown = cooldown
        # Estimate uses left (rough heuristic: assume 50 uses baseline)
        self.estimated_uses_left = max(0.0, 1.0 - (self.total_harvests / 50.0))


class ExtractorMemory:
    """Tracks all discovered extractors."""

    def __init__(self):
        # Map from resource type to list of extractors
        self._extractors: Dict[str, List[ExtractorInfo]] = defaultdict(list)
        # Map from position to extractor (for quick lookup)
        self._by_position: Dict[Tuple[int, int], ExtractorInfo] = {}

    def add_extractor(self, pos: Tuple[int, int], resource_type: str, station_name: str):
        """Add newly discovered extractor."""
        if pos in self._by_position:
            logger.debug(f"[Phase1] Extractor at {pos} already in memory")
            return  # Already known

        extractor = ExtractorInfo(position=pos, resource_type=resource_type, station_name=station_name)
        self._extractors[resource_type].append(extractor)
        self._by_position[pos] = extractor
        logger.info(f"Discovered {resource_type} extractor at {pos}")
        logger.info(f"[Phase1] Memory: {len(self._extractors[resource_type])} {resource_type} total")

    def get_by_type(self, resource_type: str) -> List[ExtractorInfo]:
        """Get all extractors of a given type."""
        return self._extractors[resource_type]

    def get_at_position(self, pos: Tuple[int, int]) -> Optional[ExtractorInfo]:
        """Get extractor at specific position."""
        return self._by_position.get(pos)

    def find_best_extractor(
        self, resource_type: str, current_pos: Tuple[int, int], current_step: int, hyperparams: Hyperparameters
    ) -> Optional[ExtractorInfo]:
        """Find best extractor considering distance, availability, efficiency."""
        extractors = self.get_by_type(resource_type)
        if not extractors:
            return None

        # Filter out depleted and unavailable
        candidates = [e for e in extractors if not e.is_depleted() and e.is_available(current_step)]

        if not candidates:
            # All on cooldown or depleted - return None to trigger exploration
            return None

        # Score each candidate
        def score_extractor(e: ExtractorInfo) -> float:
            # Distance cost (Manhattan distance)
            dist = abs(e.position[0] - current_pos[0]) + abs(e.position[1] - current_pos[1])
            distance_score = 1.0 / (1.0 + dist)  # Closer is better

            # Efficiency bonus
            avg_out = e.avg_output()
            if avg_out > 0 and hyperparams.track_efficiency:
                # Higher output is better
                efficiency_score = avg_out / 50.0  # Normalize
            else:
                efficiency_score = 0.5  # Neutral if unknown

            # Combine with weights
            if hyperparams.prefer_nearby:
                total_score = (
                    1.0 - hyperparams.efficiency_weight
                ) * distance_score + hyperparams.efficiency_weight * efficiency_score
            else:
                total_score = efficiency_score  # Only consider efficiency

            return total_score

        # Return highest scoring extractor
        return max(candidates, key=score_extractor)

    def count_available(self, resource_type: str, current_step: int) -> int:
        """Count how many extractors of this type are available."""
        return sum(1 for e in self.get_by_type(resource_type) if not e.is_depleted() and e.is_available(current_step))


class GamePhase(Enum):
    GATHER_GERMANIUM = "gather_germanium"
    GATHER_SILICON = "gather_silicon"
    GATHER_CARBON = "gather_carbon"
    GATHER_OXYGEN = "gather_oxygen"
    ASSEMBLE_HEART = "assemble_heart"
    DEPOSIT_HEART = "deposit_heart"
    RECHARGE = "recharge"


@dataclass
class AgentState:
    current_phase: GamePhase = GamePhase.GATHER_GERMANIUM
    current_glyph: str = "default"

    # Inventory
    carbon: int = 0
    oxygen: int = 0
    germanium: int = 0
    silicon: int = 0
    energy: int = 100
    heart: int = 0

    # Strategy tracking
    hearts_assembled: int = 0
    wait_counter: int = 0
    just_deposited: bool = False  # set true on successful deposit to escape the chest tile

    # Position tracking (absolute grid coordinates)
    agent_row: int = -1
    agent_col: int = -1

    # Phase 1: Home base tracking
    home_base_row: int = -1  # Remember spawn location
    home_base_col: int = -1

    # Misc
    step_count: int = 0
    last_heart: int = 0
    stuck_counter: int = 0
    last_reward: float = 0.0
    total_reward: float = 0.0


# ===============================
# Implementation
# ===============================
class ScriptedAgentPolicyImpl(StatefulPolicyImpl[AgentState]):
    """Scripted policy with visual discovery, frontier exploration, and detailed logging."""

    # ---- Constants & thresholds ----
    RECHARGE_START = 65  # enter recharge when below this (Phase 1: increased for exploration)
    RECHARGE_STOP = 90  # stay in recharge until at least this
    CARBON_REQ = 20
    OXYGEN_REQ = 20
    SILICON_REQ = 50
    ENERGY_REQ = 20

    HEART_FEATURE_NAME = "inv:heart"
    HEART_SENTINEL_FIRST_FIELD = 85  # 0x55

    # Observation window size
    OBS_HEIGHT = 11
    OBS_WIDTH = 11
    OBS_HEIGHT_RADIUS = OBS_HEIGHT // 2
    OBS_WIDTH_RADIUS = OBS_WIDTH // 2

    # Occupancy values
    OCC_UNKNOWN, OCC_FREE, OCC_WALL = 0, 1, 2

    def __init__(self, env: MettaGridEnv):
        self._env = env

        # Action / object names
        self._action_names: List[str] = env.action_names
        self._object_type_names: List[str] = env.object_type_names

        # Lookups
        self._action_lookup: Dict[str, int] = {name: i for i, name in enumerate(self._action_names)}
        obs_features = env.observation_features
        self._feature_name_to_id: Dict[str, int] = {f.name: f.id for f in obs_features.values()}

        # Vibes/glyphs
        from cogames.cogs_vs_clips.vibes import VIBES

        self._glyph_name_to_id: Dict[str, int] = {vibe.name: idx for idx, vibe in enumerate(VIBES)}

        # Stations → glyphs
        self._station_to_glyph: Dict[str, str] = {
            "charger": "charger",
            "carbon_extractor": "carbon",
            "oxygen_extractor": "oxygen",
            "germanium_extractor": "germanium",
            "silicon_extractor": "silicon",
            "assembler": "heart",  # assembling requires heart glyph
            "chest": "chest",
        }

        # Phase → desired station
        self._phase_to_station: Dict[GamePhase, str] = {
            GamePhase.GATHER_GERMANIUM: "germanium_extractor",
            GamePhase.GATHER_SILICON: "silicon_extractor",
            GamePhase.GATHER_CARBON: "carbon_extractor",
            GamePhase.GATHER_OXYGEN: "oxygen_extractor",
            GamePhase.ASSEMBLE_HEART: "assembler",
            GamePhase.DEPOSIT_HEART: "chest",
            GamePhase.RECHARGE: "charger",
        }

        # Build type_id -> station name mapping for visual discovery
        self._type_id_to_station: Dict[int, str] = {}
        self._wall_type_id: Optional[int] = None
        for type_id, name in enumerate(env.object_type_names):
            if name == "wall":
                self._wall_type_id = type_id
            elif name and not name.startswith("agent"):
                self._type_id_to_station[type_id] = name

        # Map geometry
        self._map_height = env.c_env.map_height
        self._map_width = env.c_env.map_width

        # Incremental knowledge - NO OMNISCIENCE
        self._station_positions: Dict[str, Tuple[int, int]] = {}  # discovered stations (legacy)
        self._wall_positions: set[Tuple[int, int]] = set()  # learned walls
        self._visited_cells: set[Tuple[int, int]] = set()

        # Occupancy grid: 0=unknown, 1=free, 2=wall
        self._occ = [[self.OCC_UNKNOWN for _ in range(self._map_width)] for _ in range(self._map_height)]

        # Movement tracking
        self._prev_pos: Optional[Tuple[int, int]] = None
        self._last_action_idx: Optional[int] = None
        self._last_attempt_was_use: bool = False

        # === NEW: Phase 1 Enhancements ===
        # Hyperparameters (can be tuned per experiment)
        self.hyperparams = Hyperparameters()

        # Extractor memory system
        self.extractor_memory = ExtractorMemory()

        # Station type to resource type mapping
        self._station_to_resource_type: Dict[str, str] = {
            "carbon_extractor": "carbon",
            "oxygen_extractor": "oxygen",
            "germanium_extractor": "germanium",
            "silicon_extractor": "silicon",
            "charger": "charger",
        }

        # Energy regen rate (will be detected from observations)
        self._energy_regen_rate: float = 1.0

        # Track previous inventory to detect changes
        self._prev_inventory: Dict[str, int] = {}

        # Movement actions
        self._MOVE_N = self._action_lookup.get("move_north", -1)
        self._MOVE_S = self._action_lookup.get("move_south", -1)
        self._MOVE_E = self._action_lookup.get("move_east", -1)
        self._MOVE_W = self._action_lookup.get("move_west", -1)
        self._MOVE_SET = {self._MOVE_N, self._MOVE_S, self._MOVE_E, self._MOVE_W}

        # Loop detection (cycle breaker)
        self._loop_window: List[Tuple[int, int, int]] = []  # (r,c,action_idx or -1)
        self._loop_window_max = 20

        # Simple sweep memory
        self._recent_positions: List[Tuple[int, int]] = []
        self._max_recent_positions: int = 10

        logger.info("Scripted agent initialized with visual discovery + frontier exploration")
        logger.info(f"Map size: {self._map_height}x{self._map_width}")
        logger.info(
            "Inv feature IDs: "
            + str(
                [
                    (k, self._feature_name_to_id.get(k))
                    for k in ("inv:carbon", "inv:oxygen", "inv:germanium", "inv:silicon", "inv:energy", "inv:heart")
                ]
            )
        )

    # ========== NEW: Phase 1 Helper Methods ==========

    def _can_reach_safely(
        self, target_pos: Tuple[int, int], state: AgentState, task_energy: int = 0, is_recharge: bool = False
    ) -> bool:
        """Check if agent has enough energy to reach target and complete task.

        Args:
            target_pos: Destination position
            state: Current agent state
            task_energy: Energy needed at destination (e.g., 50 for silicon)
            is_recharge: If True, this is a trip to a charger (one-way, smaller buffer)

        Returns:
            True if energy sufficient with buffer
        """
        if state.agent_row == -1:
            return True  # Position unknown, assume OK

        # Calculate Manhattan distance
        distance = abs(target_pos[0] - state.agent_row) + abs(target_pos[1] - state.agent_col)

        if is_recharge:
            # One-way trip to charger
            energy_needed = distance + 5  # Small buffer
        else:
            # For gathering: outbound + task + return
            # With passive regen, net cost per step is ~1
            # Use minimal buffer for high-energy tasks (silicon)
            net_travel_cost = (distance * 2) * 1  # Round trip
            if task_energy >= 50:  # Silicon
                buffer = 5  # Minimal buffer, trust regen on return
            else:
                buffer = max(5, min(15, distance // 2))  # Modest buffer
            energy_needed = net_travel_cost + task_energy + buffer

        return state.energy >= energy_needed

    def _update_extractor_after_use(
        self, pos: Tuple[int, int], state: AgentState, resource_gained: int, resource_type: str
    ):
        """Update extractor info after using it."""
        extractor = self.extractor_memory.get_at_position(pos)
        if not extractor:
            return

        # Determine cooldown based on resource type
        cooldown_map = {
            "carbon": 10,
            "oxygen": 100,
            "germanium": 0,
            "silicon": 0,
            "charger": 10,
        }
        cooldown = cooldown_map.get(resource_type, 10)

        extractor.update_after_use(resource_gained, state.step_count, cooldown)

        if self.hyperparams.track_efficiency:
            avg = extractor.avg_output()
            if avg > 0:
                logger.debug(
                    f"Used {resource_type} at {pos}: {resource_gained} output "
                    f"(avg: {avg:.1f}, uses: {extractor.total_harvests})"
                )

    def _detect_inventory_changes(self, state: AgentState) -> Dict[str, int]:
        """Detect what changed in inventory since last step."""
        current = {
            "carbon": state.carbon,
            "oxygen": state.oxygen,
            "germanium": state.germanium,
            "silicon": state.silicon,
            "energy": state.energy,
        }

        changes = {}
        for resource, amount in current.items():
            prev = self._prev_inventory.get(resource, amount)
            if amount != prev:
                changes[resource] = amount - prev

        # Update for next time
        self._prev_inventory = current.copy()
        return changes

    def _find_best_extractor_for_phase(self, phase: GamePhase, state: AgentState) -> Optional[Tuple[int, int]]:
        """Find best extractor for current gathering phase using extractor memory.

        Returns position of best extractor, or None if should explore.
        """
        # Map phase to resource type
        phase_to_resource = {
            GamePhase.GATHER_CARBON: "carbon",
            GamePhase.GATHER_OXYGEN: "oxygen",
            GamePhase.GATHER_GERMANIUM: "germanium",
            GamePhase.GATHER_SILICON: "silicon",
            GamePhase.RECHARGE: "charger",
        }

        resource_type = phase_to_resource.get(phase)
        if not resource_type:
            return None

        # Get task energy (silicon needs 50 energy)
        task_energy = 50 if resource_type == "silicon" else 0

        # Debug: Log discovered extractors
        all_extractors = self.extractor_memory.get_by_type(resource_type)
        logger.info(f"[Phase1] Finding {resource_type}: {len(all_extractors)} total discovered")

        # Find best available extractor
        current_pos = (state.agent_row, state.agent_col)
        best = self.extractor_memory.find_best_extractor(resource_type, current_pos, state.step_count, self.hyperparams)

        if best is None:
            # No available extractors - check if we should wait for cooldowns
            if self.hyperparams.enable_cooldown_waiting and len(all_extractors) > 0:
                # Find extractor with shortest cooldown remaining
                extractors_on_cooldown = [
                    e for e in all_extractors if not e.is_available(state.step_count) and not e.is_depleted()
                ]

                if extractors_on_cooldown:
                    # Sort by cooldown remaining
                    nearest_cooldown = min(
                        extractors_on_cooldown, key=lambda e: self.cooldown_remaining(e, state.step_count)
                    )
                    cooldown_time = self.cooldown_remaining(nearest_cooldown, state.step_count)

                    # If cooldown is reasonable, wait near it
                    if cooldown_time <= self.hyperparams.max_cooldown_wait:
                        logger.info(
                            f"[Phase1] Waiting for {resource_type} at {nearest_cooldown.position} "
                            f"(cooldown: {cooldown_time} turns)"
                        )
                        return nearest_cooldown.position
                    else:
                        logger.info(
                            f"[Phase1] Cooldown too long ({cooldown_time} > {self.hyperparams.max_cooldown_wait}), "
                            f"will explore for new {resource_type}"
                        )

            # No available extractors - need to explore
            logger.info(
                f"[Phase1] No available {resource_type} extractors "
                f"(found {len(all_extractors)} but all depleted/on cooldown)"
            )
            return None

        distance = abs(best.position[0] - state.agent_row) + abs(best.position[1] - state.agent_col)
        logger.info(f"[Phase1] Best {resource_type} extractor: {best.position}, distance={distance}")

        # Check if we have energy to reach it
        is_charger = resource_type == "charger"
        if not self._can_reach_safely(best.position, state, task_energy, is_recharge=is_charger):
            # Log why we can't reach it (using same calculation as _can_reach_safely)
            if is_charger:
                energy_needed = distance + 5
            else:
                net_travel_cost = (distance * 2) * 1
                if task_energy >= 50:  # Silicon
                    buffer = 5
                else:
                    buffer = max(5, min(15, distance // 2))
                energy_needed = net_travel_cost + task_energy + buffer
            logger.warning(
                f"[Phase1] Not enough energy to reach {resource_type} at {best.position} "
                f"(need ~{energy_needed}, have {state.energy})"
            )
            return None

        logger.info(f"[Phase1] ✓ Selected {resource_type} at {best.position}")
        return best.position

    # ---------- Policy Interface ----------
    def agent_state(self) -> AgentState:
        return AgentState()

    def step_with_state(
        self, obs: MettaGridObservation, state: Optional[AgentState]
    ) -> tuple[MettaGridAction, Optional[AgentState]]:
        """Main policy step: update knowledge, select phase, choose an action."""
        if state is None:
            state = self.agent_state()
        state.step_count += 1

        # Update world & agent state from observation
        self._update_inventory(obs, state)
        self._update_agent_position(state)
        self._update_rewards(obs, state)

        # Phase 1: Remember spawn location as home base
        if state.home_base_row == -1 and state.agent_row >= 0:
            state.home_base_row = state.agent_row
            state.home_base_col = state.agent_col
            logger.info(f"[Phase1] Home base set to ({state.home_base_row}, {state.home_base_col})")

        # Mark current as free and update map from observation
        self._mark_cell(state.agent_row, state.agent_col, self.OCC_FREE)
        self._discover_stations_from_observation(obs, state)
        self._update_wall_knowledge(state)

        # === NEW: Track extractor usage ===
        # Detect inventory changes to update extractor stats
        inv_changes = self._detect_inventory_changes(state)
        if inv_changes and state.agent_row >= 0:
            # Check if we just used an extractor at an adjacent position
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                check_pos = (state.agent_row + dr, state.agent_col + dc)
                extractor = self.extractor_memory.get_at_position(check_pos)
                if extractor:
                    # Check if the inventory change matches the extractor type
                    resource = extractor.resource_type
                    if resource in inv_changes and inv_changes[resource] > 0:
                        self._update_extractor_after_use(check_pos, state, inv_changes[resource], resource)
                        break

        # Track visited
        if state.agent_row >= 0 and state.agent_col >= 0:
            self._visited_cells.add((state.agent_row, state.agent_col))

        # Deposit detection
        if (state.last_heart > 0 and state.heart == 0) or (
            state.current_phase == GamePhase.DEPOSIT_HEART and state.last_reward > 0
        ):
            state.wait_counter = 0
            state.current_phase = GamePhase.GATHER_GERMANIUM
            state.just_deposited = True
            logger.info(f"Step {state.step_count}: Heart deposited -> restarting gather phase")

        # Decide phase & act
        state.current_phase = self._determine_phase(state)
        action_idx = self._execute_phase(state)

        # Simple loop breaking: if we repeat the same (pos, action) too often, try alternatives
        self._apply_cycle_breaker(state, action_idx)

        # Bookkeeping
        self._last_action_idx = action_idx
        self._prev_pos = (state.agent_row, state.agent_col)
        state.last_heart = state.heart

        return dtype_actions.type(action_idx), state

    # ---------- Observation → Knowledge ----------
    def _discover_stations_from_observation(self, obs: MettaGridObservation, state: AgentState) -> None:
        """Discover stations/walls from vision and update occupancy grid.

        Rules:
        - Walls: mark OCC_WALL
        - Stations: mark OCC_WALL (you use them from adjacent cells)
        - Other objects (not agents/walls/stations): mark OCC_FREE
        - Always mark current agent position as OCC_FREE
        """
        if state.agent_row == -1:
            return

        type_id_feature = self._feature_name_to_id.get("type_id", 0)

        for tok in obs:
            if self._to_int(tok[1]) != type_id_feature:
                continue

            # Decode local coords (hi nibble=row, lo nibble=col)
            packed = self._to_int(tok[0])
            obs_r = packed >> 4
            obs_c = packed & 0x0F
            type_id = self._to_int(tok[2])

            # Convert to absolute map coords
            map_r = obs_r - self.OBS_HEIGHT_RADIUS + state.agent_row
            map_c = obs_c - self.OBS_WIDTH_RADIUS + state.agent_col
            if not self._is_valid_position(map_r, map_c):
                continue

            if type_id == self._wall_type_id:
                # Wall (unwalkable)
                self._mark_cell(map_r, map_c, self.OCC_WALL)
                self._wall_positions.add((map_r, map_c))
                continue

            if type_id in self._type_id_to_station:
                # Station (unwalkable). Remember first seen location.
                station_name = self._type_id_to_station[type_id]
                self._mark_cell(map_r, map_c, self.OCC_WALL)
                pos = (map_r, map_c)
                if station_name not in self._station_positions:
                    self._station_positions[station_name] = pos
                    logger.info(f"Discovered {station_name} at {pos}")

                # Also add to extractor memory if it's a resource extractor
                resource_type = self._station_to_resource_type.get(station_name)
                if resource_type:
                    self.extractor_memory.add_extractor(pos, resource_type, station_name)
                else:
                    # Debug: station name didn't match our mapping
                    logger.debug(f"[Phase1] Station '{station_name}' not in resource_type mapping")
                continue

            # Non-agent, non-wall, non-station object: treat as free
            if not self._object_type_names[type_id].startswith("agent"):
                self._mark_cell(map_r, map_c, self.OCC_FREE)

        # Ensure current cell is free
        self._mark_cell(state.agent_row, state.agent_col, self.OCC_FREE)

    def _update_wall_knowledge(self, state: AgentState) -> None:
        """Update occupancy based on movement results.

        - If we moved successfully, mark new cell as free
        - If we tried to move but didn't, mark intended target as wall
        """
        if not self._prev_pos or self._last_action_idx not in self._MOVE_SET:
            return

        cur = (state.agent_row, state.agent_col)
        if cur != self._prev_pos:
            # Moved: mark new cell free
            self._mark_cell(cur[0], cur[1], self.OCC_FREE)
            return

        # Didn't move: intended target is blocked
        dr, dc = self._action_to_dir(self._last_action_idx)
        if dr is None or dc is None:
            return

        wr, wc = self._prev_pos[0] + dr, self._prev_pos[1] + dc
        if not self._is_valid_position(wr, wc):
            return

        # Mark as wall (could be a wall or a station; usable from adjacency)
        if self._occ[wr][wc] != self.OCC_WALL:
            logger.info(f"Marking blocked cell at ({wr},{wc})")
        self._wall_positions.add((wr, wc))
        self._occ[wr][wc] = self.OCC_WALL

    # ---------- Phase & Action Selection ----------
    def _determine_phase(self, state: AgentState) -> GamePhase:
        """Choose the current high-level phase."""
        if state.heart > 0:
            return GamePhase.DEPOSIT_HEART

        germ_needed = 5 if state.hearts_assembled == 0 else max(2, 5 - state.hearts_assembled)

        # Recharge hysteresis
        if state.current_phase == GamePhase.RECHARGE and state.energy < self.RECHARGE_STOP:
            return GamePhase.RECHARGE
        if state.energy < self.RECHARGE_START:
            return GamePhase.RECHARGE

        # Check if we have all resources for assembly
        has_all_resources = (
            state.germanium >= germ_needed
            and state.carbon >= self.CARBON_REQ
            and state.oxygen >= self.OXYGEN_REQ
            and state.silicon >= self.SILICON_REQ
        )

        # If we have all resources but not enough energy, recharge first
        if has_all_resources and state.energy < self.ENERGY_REQ:
            logger.info(f"[Phase1] Have all resources, charging for assembly (energy={state.energy}/{self.ENERGY_REQ})")
            return GamePhase.RECHARGE

        # Assembling possible?
        if has_all_resources and state.energy >= self.ENERGY_REQ:
            return GamePhase.ASSEMBLE_HEART

        # Resource priorities
        if state.germanium < germ_needed:
            return GamePhase.GATHER_GERMANIUM
        if state.silicon < self.SILICON_REQ:
            return GamePhase.GATHER_SILICON
        if state.carbon < self.CARBON_REQ:
            return GamePhase.GATHER_CARBON
        if state.oxygen < self.OXYGEN_REQ:
            return GamePhase.GATHER_OXYGEN

        return GamePhase.GATHER_GERMANIUM

    def _execute_phase(self, state: AgentState) -> int:
        """Convert phase to a concrete action (move or glyph change)."""
        station = self._phase_to_station.get(state.current_phase)
        if not station:
            return self._action_lookup.get("noop", 0)

        # After deposit, if standing on chest, immediately switch to gather to move away
        if state.just_deposited:
            gather_station = self._phase_to_station[GamePhase.GATHER_GERMANIUM]
            if state.agent_row != -1 and "chest" in self._station_positions:
                cr, cc = self._station_positions["chest"]
                if (state.agent_row, state.agent_col) == (cr, cc):
                    state.current_phase = GamePhase.GATHER_GERMANIUM
                    station = gather_station
                else:
                    state.just_deposited = False
            else:
                state.current_phase = GamePhase.GATHER_GERMANIUM
                station = gather_station

        # Glyph switching
        needed_glyph = self._station_to_glyph.get(station, "default")
        if state.current_glyph != needed_glyph:
            state.current_glyph = needed_glyph
            state.wait_counter = 0
            glyph_id = self._glyph_name_to_id.get(needed_glyph, 0)
            return self._action_lookup.get(f"change_glyph_{glyph_id}", self._action_lookup.get("noop", 0))

        # === NEW: Use extractor memory for gathering phases ===
        is_gathering_phase = state.current_phase in [
            GamePhase.GATHER_CARBON,
            GamePhase.GATHER_OXYGEN,
            GamePhase.GATHER_GERMANIUM,
            GamePhase.GATHER_SILICON,
            GamePhase.RECHARGE,
        ]

        target_pos = None
        if is_gathering_phase and state.agent_row != -1:
            # Try to find best extractor using memory
            target_pos = self._find_best_extractor_for_phase(state.current_phase, state)

            if target_pos is None:
                # No available extractors - explore to find more
                self._last_attempt_was_use = False
                logger.info(f"[Phase1] {state.current_phase.value}: No available extractors, exploring")
                plan = self._plan_to_frontier_action(state)
                return plan if plan is not None else self._explore_simple(state)
            else:
                logger.info(f"[Phase1] {state.current_phase.value}: Targeting extractor at {target_pos}")

        # Fallback to old single-station logic for non-gathering phases or if extractor memory failed
        if target_pos is None:
            # Old logic: use legacy _station_positions
            if station in self._station_positions and state.agent_row != -1:
                target_pos = self._station_positions[station]
            else:
                # Unknown station: explore
                self._last_attempt_was_use = False
                plan = self._plan_to_frontier_action(state)
                return plan if plan is not None else self._explore_simple(state)

        # Now navigate to target_pos
        tr, tc = target_pos
        dr, dc = tr - state.agent_row, tc - state.agent_col
        mdist = abs(dr) + abs(dc)

        # Adjacent: attempt to step into the station (use)
        if mdist == 1:
            self._last_attempt_was_use = True
            return self._step_toward(dr, dc)

        # Otherwise: path to a non-blocked neighbor of the station (use A* for long paths)
        adj = [(nr, nc) for nr, nc in self._neighbors4(tr, tc) if self._occ[nr][nc] != self.OCC_WALL]
        if adj:
            adj.sort(key=lambda p: abs(p[0] - state.agent_row) + abs(p[1] - state.agent_col))
            for goal in adj:
                # Use intelligent pathfinding (A* for long distances, BFS for short)
                step = self._choose_pathfinding((state.agent_row, state.agent_col), goal, optimistic=True)
                if step is not None:
                    nr, nc = step
                    self._last_attempt_was_use = False
                    return self._step_toward(nr - state.agent_row, nc - state.agent_col)

        # Not reachable yet: reveal more map
        self._last_attempt_was_use = False
        plan = self._plan_to_frontier_action(state)
        return plan if plan is not None else self._explore_simple(state)

    # ---------- Exploration / Frontier ----------
    def _plan_to_frontier_action(self, state: AgentState) -> Optional[int]:
        """Plan one step toward the nearest frontier (unknown adjacent to free)."""
        target = self._choose_frontier(state)
        if not target:
            return None

        tr, tc = target
        sr, sc = state.agent_row, state.agent_col

        # If adjacent to frontier, step into it directly
        if abs(tr - sr) + abs(tc - sc) == 1:
            return self._step_toward(tr - sr, tc - sc)

        # Else path to a free neighbor of the frontier (excluding our current cell)
        candidates = [
            (nr, nc)
            for nr, nc in self._neighbors4(tr, tc)
            if self._occ[nr][nc] == self.OCC_FREE and (nr, nc) != (sr, sc)
        ]
        if not candidates:
            return None

        candidates.sort(key=lambda p: abs(p[0] - sr) + abs(p[1] - sc))
        for goal in candidates:
            step = self._bfs_next_step_occ((sr, sc), goal)
            if step is not None:
                dr, dc = step[0] - sr, step[1] - sc
                return self._step_toward(dr, dc)
        return None

    def _choose_frontier(self, state: AgentState) -> Optional[Tuple[int, int]]:
        """Pick the nearest frontier by BFS distance over known-free cells."""
        if state.agent_row < 0:
            return None

        start = (state.agent_row, state.agent_col)
        fronts = set(self._compute_frontiers())
        if not fronts:
            return None

        q = deque([start])
        seen = {start}
        while q:
            r, c = q.popleft()

            # If a frontier is adjacent, choose it
            for nr, nc in self._neighbors4(r, c):
                if (nr, nc) in fronts:
                    return (nr, nc)

            # Otherwise expand over known-free cells
            for nr, nc in self._neighbors4(r, c):
                if (nr, nc) in seen or self._occ[nr][nc] != self.OCC_FREE:
                    continue
                seen.add((nr, nc))
                q.append((nr, nc))
        return None

    def _compute_frontiers(self) -> List[Tuple[int, int]]:
        """Unknown cells that are 4-adjacent to a known free cell."""
        res: List[Tuple[int, int]] = []
        for r in range(self._map_height):
            for c in range(self._map_width):
                if self._occ[r][c] != self.OCC_UNKNOWN:
                    continue
                for nr, nc in self._neighbors4(r, c):
                    if self._occ[nr][nc] == self.OCC_FREE:
                        res.append((r, c))
                        break
        return res

    # ---------- Pathfinding ----------
    def _bfs_next_step_occ(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """BFS through known-free cells only; returns next cell on path or None."""
        return self._bfs_next_step(start, goal, optimistic=False)

    def _bfs_next_step_optimistic(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """BFS treating unknown cells as walkable; avoids only known walls."""
        return self._bfs_next_step(start, goal, optimistic=True)

    def _bfs_next_step(
        self, start: Tuple[int, int], goal: Tuple[int, int], optimistic: bool
    ) -> Optional[Tuple[int, int]]:
        """Grid BFS; return next cell toward goal or None."""
        if start == goal:
            return start

        q = deque([start])
        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}

        while q:
            r, c = q.popleft()
            for nr, nc in self._neighbors4(r, c):
                if (nr, nc) in parent:
                    continue
                if not self._is_cell_passable(nr, nc, optimistic):
                    continue

                parent[(nr, nc)] = (r, c)
                if (nr, nc) == goal:
                    return self._reconstruct_first_step(parent, start, goal)
                q.append((nr, nc))
        return None

    def _reconstruct_first_step(
        self, parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]], start: Tuple[int, int], goal: Tuple[int, int]
    ) -> Tuple[int, int]:
        """Reconstruct the first step from start towards goal using parent pointers."""
        step = goal
        while parent[step] != start:
            step = parent[step]  # type: ignore[assignment]
        return step

    def _is_cell_passable(self, r: int, c: int, optimistic: bool) -> bool:
        """Check if a cell is passable for BFS."""
        cell_state = self._occ[r][c]
        return (cell_state != self.OCC_WALL) if optimistic else (cell_state == self.OCC_FREE)

    def _astar_next_step(
        self, start: Tuple[int, int], goal: Tuple[int, int], optimistic: bool = True
    ) -> Optional[Tuple[int, int]]:
        """A* pathfinding with Manhattan distance heuristic.

        Much faster than BFS on large maps with complex paths.
        Returns next cell toward goal or None if no path exists.
        """
        import heapq

        if start == goal:
            return start

        def heuristic(pos: Tuple[int, int]) -> int:
            """Manhattan distance heuristic."""
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

        # Priority queue: (f_score, g_score, position)
        open_set = [(heuristic(start), 0, start)]
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        g_score: Dict[Tuple[int, int], int] = {start: 0}

        while open_set:
            current_f, current_g, current = heapq.heappop(open_set)

            # If we reached goal, reconstruct path
            if current == goal:
                return self._reconstruct_first_step(came_from, start, goal)

            # Skip if we've found a better path to this node already
            if current_g > g_score.get(current, float("inf")):
                continue

            # Explore neighbors
            for next_pos in self._neighbors4(*current):
                if not self._is_cell_passable(next_pos[0], next_pos[1], optimistic):
                    continue

                tentative_g = current_g + 1
                if next_pos not in g_score or tentative_g < g_score[next_pos]:
                    g_score[next_pos] = tentative_g
                    f_score = tentative_g + heuristic(next_pos)
                    heapq.heappush(open_set, (f_score, tentative_g, next_pos))
                    came_from[next_pos] = current

        return None  # No path found

    def _choose_pathfinding(
        self, start: Tuple[int, int], goal: Tuple[int, int], optimistic: bool = True
    ) -> Optional[Tuple[int, int]]:
        """Intelligently choose between BFS and A* based on distance.

        - Use BFS for short distances (faster for small search spaces)
        - Use A* for long distances (much faster on large mazes)
        """
        if not self.hyperparams.use_astar:
            return self._bfs_next_step(start, goal, optimistic)

        # Calculate Manhattan distance
        distance = abs(goal[0] - start[0]) + abs(goal[1] - start[1])

        # Use A* for long paths, BFS for short ones
        if distance >= self.hyperparams.astar_threshold:
            logger.debug(f"Using A* for distance {distance} (>= {self.hyperparams.astar_threshold})")
            return self._astar_next_step(start, goal, optimistic)
        else:
            return self._bfs_next_step(start, goal, optimistic)

    def cooldown_remaining(self, extractor: ExtractorInfo, current_step: int) -> int:
        """Calculate remaining cooldown turns for an extractor."""
        if extractor.last_used_step < 0:
            return 0  # Never used
        elapsed = current_step - extractor.last_used_step
        remaining = max(0, extractor.estimated_cooldown - elapsed)
        return remaining

    # ---------- Exploration fallback ----------
    def _explore_simple(self, state: AgentState) -> int:
        """Boustrophedon sweep with simple loop avoidance."""
        if state.agent_row == -1:
            return self._action_lookup.get("noop", 0)

        # Track recent positions for loop avoidance
        cur = (state.agent_row, state.agent_col)
        if not self._recent_positions or self._recent_positions[-1] != cur:
            self._recent_positions.append(cur)
            if len(self._recent_positions) > self._max_recent_positions:
                self._recent_positions.pop(0)

        # Horizontal sweep based on row parity
        row_parity = state.agent_row % 2
        preferred_dir = self._MOVE_E if row_parity == 0 else self._MOVE_W

        dr, dc = self._action_to_dir(preferred_dir)
        nr, nc = state.agent_row + (dr or 0), state.agent_col + (dc or 0)
        if preferred_dir in self._MOVE_SET and self._is_valid_position(nr, nc) and (nr, nc) not in self._wall_positions:
            return preferred_dir

        # Try moving down a row
        down_r, down_c = state.agent_row + 1, state.agent_col
        if self._is_valid_position(down_r, down_c) and (down_r, down_c) not in self._wall_positions:
            return self._MOVE_S if self._MOVE_S != -1 else self._action_lookup.get("noop", 0)

        # Otherwise pick any reasonable alternative, preferring not-recent cells
        alt = self._find_best_exploration_direction(state)
        return (
            alt if alt is not None else (preferred_dir if preferred_dir != -1 else self._action_lookup.get("noop", 0))
        )

    def _find_best_exploration_direction(self, state: AgentState) -> Optional[int]:
        """Pick a direction toward an in-bounds, not-wall cell; prefer not-recent."""
        options = [
            (self._MOVE_N, (-1, 0)),
            (self._MOVE_S, (1, 0)),
            (self._MOVE_E, (0, 1)),
            (self._MOVE_W, (0, -1)),
        ]
        best_action, best_score = None, -1
        for action, (dr, dc) in options:
            nr, nc = state.agent_row + dr, state.agent_col + dc
            if not self._is_valid_position(nr, nc):
                continue
            pos = (nr, nc)
            if pos in self._wall_positions:
                continue
            score = 10 if pos not in self._recent_positions else self._recent_positions.index(pos)
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    # ---------- Utilities ----------
    @staticmethod
    def _to_int(x) -> int:
        """Convert various numeric types to int."""
        if isinstance(x, int):
            return x

        # Handle arrays/sequences - always take first element
        if hasattr(x, "__len__"):
            # Multi-element array - take first
            if len(x) > 1:
                return int(x[0])
            # Single element array
            elif len(x) == 1:
                # Recursively convert the single element
                return ScriptedAgentPolicyImpl._to_int(x[0])

        # Handle numpy/torch scalars
        if hasattr(x, "item"):
            return int(x.item())

        return int(x)

    def _is_valid_position(self, r: int, c: int) -> bool:
        """Check map bounds."""
        return 0 <= r < self._map_height and 0 <= c < self._map_width

    def _mark_cell(self, r: int, c: int, cell_type: int) -> None:
        """Mark occupancy if in-bounds."""
        if self._is_valid_position(r, c):
            self._occ[r][c] = cell_type

    def _neighbors4(self, r: int, c: int) -> List[Tuple[int, int]]:
        """4-connected neighbors inside bounds."""
        res: List[Tuple[int, int]] = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if self._is_valid_position(nr, nc):
                res.append((nr, nc))
        return res

    def _action_to_dir(self, action_idx: int) -> Tuple[Optional[int], Optional[int]]:
        if action_idx == self._MOVE_N:
            return -1, 0
        if action_idx == self._MOVE_S:
            return 1, 0
        if action_idx == self._MOVE_E:
            return 0, 1
        if action_idx == self._MOVE_W:
            return 0, -1
        return None, None

    def _step_toward(self, dr: int, dc: int) -> int:
        if dr > 0:
            return self._MOVE_S
        if dr < 0:
            return self._MOVE_N
        if dc > 0:
            return self._MOVE_E
        if dc < 0:
            return self._MOVE_W
        return self._action_lookup.get("noop", 0)

    # ---------- Inventory / Rewards ----------
    def _update_agent_position(self, state: AgentState) -> None:
        """Read agent's absolute position from env.c_env.grid_objects."""
        try:
            for _obj_id, obj in self._env.c_env.grid_objects().items():
                if obj.get("agent_id") == 0:
                    state.agent_row = obj.get("r", -1)
                    state.agent_col = obj.get("c", -1)
                    break
        except Exception as e:
            logger.debug(f"Could not update agent position: {e}")

    def _update_rewards(self, obs: MettaGridObservation, state: AgentState) -> None:
        """Accumulate per-step reward from observation."""
        try:
            fid = self._feature_name_to_id.get("global:last_reward")
            if fid is None:
                return
            for tok in obs:
                if self._to_int(tok[1]) == fid:
                    state.last_reward = float(self._to_int(tok[2]))
                    state.total_reward += state.last_reward
                    break
        except Exception:
            pass

    def _has_heart_from_obs(self, obs: MettaGridObservation) -> bool:
        """Heart present only if FIRST 'inv:heart' token's first field == 85 (0x55)."""
        heart_fid = self._feature_name_to_id.get(self.HEART_FEATURE_NAME)
        if heart_fid is None:
            return False
        first_idx = None
        for i, tok in enumerate(obs):
            if self._to_int(tok[1]) == heart_fid:
                first_idx = i
                break
        if first_idx is None:
            return False
        first_tok = obs[first_idx]
        return self._to_int(first_tok[0]) == self.HEART_SENTINEL_FIRST_FIELD

    def _read_int_feature(self, obs: MettaGridObservation, feat_name: str) -> int:
        fid = self._feature_name_to_id.get(feat_name)
        if fid is None:
            return 0
        for tok in obs:
            if self._to_int(tok[1]) == fid:
                return self._to_int(tok[2])
        return 0

    def _update_inventory(self, obs: MettaGridObservation, state: AgentState) -> None:
        """Read inventory strictly from observation tokens."""
        state.carbon = self._read_int_feature(obs, "inv:carbon")
        state.oxygen = self._read_int_feature(obs, "inv:oxygen")
        state.germanium = self._read_int_feature(obs, "inv:germanium")
        state.silicon = self._read_int_feature(obs, "inv:silicon")
        state.energy = self._read_int_feature(obs, "inv:energy")
        state.heart = 1 if self._has_heart_from_obs(obs) else 0

    # ---------- Loop breaker ----------
    def _apply_cycle_breaker(self, state: AgentState, action_idx: int) -> None:
        """If we keep repeating (pos, action), try a different direction."""
        key = (state.agent_row, state.agent_col, action_idx)
        self._loop_window.append(key)
        if len(self._loop_window) > self._loop_window_max:
            self._loop_window.pop(0)

        # If the same key appears more than 3 times in the window, perturb
        if self._loop_window.count(key) > 3:
            if state.step_count % 10 == 0:
                action_name = self._action_names[action_idx]
                logger.info(f"Stuck at ({state.agent_row},{state.agent_col}) action={action_name}, trying alternatives")
            for a in (self._MOVE_E, self._MOVE_W, self._MOVE_S, self._MOVE_N):
                if a in self._MOVE_SET and a != action_idx:
                    # Replace last action attempt with a perturbation
                    self._loop_window = []
                    self._last_action_idx = a
                    break

    # ---------- Optional logging ----------
    def _print_step_log(self, action_idx: int, state: AgentState) -> None:
        try:
            action_name = self._action_names[action_idx] if 0 <= action_idx < len(self._action_names) else "?"
            target_station = self._phase_to_station.get(state.current_phase, "unknown")
            tr, tc = self._station_positions.get(target_station, (-1, -1))
            rel = f"({tr - state.agent_row},{tc - state.agent_col})" if state.agent_row != -1 and tr != -1 else "?"
            print(
                f"Timestep : {state.step_count}, Current phase: {state.current_phase}, "
                f"action: {action_name}, target station: {target_station}, relative position: {rel}, "
                f"notes: wait={state.wait_counter}, "
                f"energy: {state.energy}, carbon: {state.carbon}, oxygen: {state.oxygen}, "
                f"germanium: {state.germanium}, silicon: {state.silicon}, heart: {state.heart}, "
                f"last_reward: {state.last_reward}, total_reward: {state.total_reward}"
            )
        except Exception as e:
            logger.debug(f"Print step log failed: {e}")


# ===============================
# Public Policy wrapper
# ===============================
class ScriptedAgentPolicy(Policy):
    """Scripted policy for training facility missions."""

    def __init__(self, env: MettaGridEnv, device=None):
        self._env = env
        self._impl = ScriptedAgentPolicyImpl(env)

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        return StatefulAgentPolicy(self._impl, agent_id)
