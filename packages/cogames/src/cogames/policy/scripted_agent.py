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
from typing import Dict, List, Optional, Set, Tuple

from cogames.policy.hyperparameters_streamlined import Hyperparameters
from cogames.policy.interfaces import AgentPolicy, Policy, StatefulAgentPolicy, StatefulPolicyImpl
from cogames.policy.navigator import Navigator
from cogames.policy.phase_controller import Context, GamePhase, create_controller
from mettagrid import MettaGridAction, MettaGridEnv, MettaGridObservation, dtype_actions

logger = logging.getLogger("cogames.policy.scripted_agent")


# ===============================
# Enums & Data
# ===============================


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

    # Properties from observations
    uses_remaining_fraction: float = 1.0  # Fraction of uses remaining (0-1, from observations)
    observed_cooldown_remaining: int = 0  # Actual cooldown remaining from observations (when in view)
    observed_converting: bool = False  # Whether station is converting/cooling down (when in view)
    is_clipped: bool = False  # Whether station is clipped (from observations)
    permanently_depleted: bool = False  # Marked as dead when remaining_uses == 0

    # Learned cooldown (inferred from observations after first use)
    learned_cooldown: Optional[int] = None  # Total cooldown duration (learned from observations)

    # Hyperparameters for access in methods
    hyperparams: Optional[Hyperparameters] = None

    def is_available(self, current_step: int, cooldown_estimate_fn) -> bool:
        """Check if extractor is available (not depleted, not on cooldown, not clipped)."""
        if self.permanently_depleted or self.is_clipped:
            return False
        rem = cooldown_estimate_fn(self, current_step)
        return rem <= 0

    def is_depleted(self) -> bool:
        """Check if extractor is likely depleted."""
        return self.permanently_depleted or self.uses_remaining_fraction <= 0.05

    def is_low(self, depletion_threshold: float | None = None) -> bool:
        """Check if extractor is running low (below threshold, should find backup)."""
        if depletion_threshold is None:
            # Use hyperparameter default if available
            if hasattr(self, "hyperparams") and self.hyperparams:
                depletion_threshold = self.hyperparams.depletion_threshold
            else:
                depletion_threshold = 0.25  # fallback default
        return self.uses_remaining_fraction <= depletion_threshold

    def avg_output(self) -> float:
        """Average output per use."""
        if self.total_harvests == 0:
            return 0.0
        return self.total_output / self.total_harvests

    def update_after_use(self, output: int, current_step: int):
        """Update stats after using extractor."""
        self.last_used_step = current_step
        self.total_harvests += 1
        self.total_output += output
        # Update uses remaining fraction (heuristic: assume 50 uses baseline if not observed)
        self.uses_remaining_fraction = max(0.0, 1.0 - (self.total_harvests / 50.0))


class ExtractorMemory:
    """Tracks all discovered extractors."""

    def __init__(self, hyperparams: Hyperparameters | None = None):
        # Map from resource type to list of extractors
        self._extractors: Dict[str, List[ExtractorInfo]] = defaultdict(list)
        # Map from position to extractor (for quick lookup)
        self._by_position: Dict[Tuple[int, int], ExtractorInfo] = {}
        # Store hyperparameters for access in methods
        self.hyperparams = hyperparams

    def add_extractor(self, pos: Tuple[int, int], resource_type: str, station_name: str) -> ExtractorInfo:
        """Add newly discovered extractor or return existing one."""
        if pos in self._by_position:
            logger.debug(f"[Phase1] Extractor at {pos} already in memory")
            return self._by_position[pos]  # Return existing

        extractor = ExtractorInfo(position=pos, resource_type=resource_type, station_name=station_name)
        # Pass hyperparameters to the extractor
        if self.hyperparams:
            extractor.hyperparams = self.hyperparams
        self._extractors[resource_type].append(extractor)
        self._by_position[pos] = extractor
        logger.info(f"Discovered {resource_type} extractor at {pos}")
        logger.info(f"[Phase1] Memory: {len(self._extractors[resource_type])} {resource_type} total")
        return extractor

    def get_by_type(self, resource_type: str) -> List[ExtractorInfo]:
        """Get all extractors of a given type."""
        return self._extractors[resource_type]

    def get_all(self) -> List[ExtractorInfo]:
        """Get all extractors of any type."""
        res: List[ExtractorInfo] = []
        for lst in self._extractors.values():
            res.extend(lst)
        return res

    def get_at_position(self, pos: Tuple[int, int]) -> Optional[ExtractorInfo]:
        """Get extractor at specific position."""
        return self._by_position.get(pos)

    def find_best_extractor(
        self, resource_type: str, current_pos: Tuple[int, int], current_step: int, cooldown_estimate_fn
    ) -> Optional[ExtractorInfo]:
        """Find best extractor considering distance, availability, efficiency."""
        extractors = self.get_by_type(resource_type)
        if not extractors:
            return None

        # Filter out depleted and unavailable
        candidates = []
        for e in extractors:
            if e.is_depleted():
                logger.debug(f"[FindExtractor] {resource_type} at {e.position} is depleted")
                continue
            if not e.is_available(current_step, cooldown_estimate_fn):
                logger.debug(
                    f"[FindExtractor] {resource_type} at {e.position} unavailable: "
                    f"cooldown_est={cooldown_estimate_fn(e, current_step)}, "
                    f"last_used={e.last_used_step}, current={current_step}"
                )
                continue
            candidates.append(e)

        if not candidates:
            # All on cooldown or depleted - return None to trigger exploration or waiting
            logger.info(f"[FindExtractor] No available {resource_type} extractors (total={len(extractors)})")
            return None

        # Score each candidate
        def score_extractor(e: ExtractorInfo) -> float:
            # Distance cost (Manhattan distance)
            dist = abs(e.position[0] - current_pos[0]) + abs(e.position[1] - current_pos[1])
            distance_score = 1.0 / (1.0 + dist)  # Closer is better

            # Efficiency bonus
            avg_out = e.avg_output()
            efficiency_score = (avg_out / 50.0) if avg_out > 0 else 0.5

            # Depletion penalty
            depletion_penalty = 0.5 if e.is_low(0.25) else 1.0  # Fixed threshold

            # Use fixed weights
            w_d = 0.7  # Distance weight
            w_e = 0.3  # Efficiency weight
            total_score = (w_d * distance_score + w_e * efficiency_score) * depletion_penalty

            # Clip avoidance bias
            if e.is_clipped:
                total_score *= 0.5  # Fixed bias

            return total_score

        # Return highest scoring extractor
        return max(candidates, key=score_extractor)


# GamePhase enum is now imported from phase_controller


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
    # Unclip items
    decoder: int = 0
    modulator: int = 0
    resonator: int = 0
    scrambler: int = 0

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

    # Phase 3: Critical station tracking
    assembler_discovered: bool = False
    chest_discovered: bool = False

    # Phase 3: Cooldown waiting
    waiting_since_step: int = -1
    wait_target: Optional[Tuple[int, int]] = None

    # Unclipping tracking
    unclip_target: Optional[Tuple[int, int]] = None  # Position of station to unclip
    # Recipe knowledge: what resource is needed to craft each unclip item
    unclip_recipes: Optional[Dict[str, str]] = None  # e.g., {"decoder": "carbon", "modulator": "oxygen"}

    # Phase tracking for stuck detection
    phase_entry_step: int = 0  # When we entered current phase
    phase_entry_inventory: Optional[Dict[str, int]] = None  # Inventory when we entered phase
    unobtainable_resources: Optional[Set[str]] = None  # Resources we've given up on (too hard to get)
    resource_gathering_start: Optional[Dict[str, int]] = None  # When we first started trying to gather each resource
    resource_progress_tracking: Optional[Dict[str, int]] = None  # Initial amount of each resource when we started
    phase_visit_count: Optional[Dict[str, int]] = None  # Count how many times we've entered each gathering phase

    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.phase_entry_inventory is None:
            self.phase_entry_inventory = {}
        if self.unobtainable_resources is None:
            self.unobtainable_resources = set()
        if self.resource_gathering_start is None:
            self.resource_gathering_start = {}
        if self.resource_progress_tracking is None:
            self.resource_progress_tracking = {}
        if self.phase_visit_count is None:
            self.phase_visit_count = {}
        if self.unclip_recipes is None:
            self.unclip_recipes = {}

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
    # Dynamic recharge thresholds (adjusted for map size) - now from hyperparams
    # RECHARGE_START_SMALL = 65  # for maps < 50x50
    # RECHARGE_START_LARGE = 45  # for maps >= 50x50 (less recharging)
    # RECHARGE_STOP_SMALL = 90
    # RECHARGE_STOP_LARGE = 75  # Recharge less on large maps

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
    OCC_UNKNOWN, OCC_FREE, OCC_OBSTACLE = 0, 1, 2

    # Planning energy model: passive regeneration per step (constant) - now from hyperparams
    # PASSIVE_REGEN_PER_STEP = 1.0

    def __init__(self, env: MettaGridEnv, hyperparams: Hyperparameters | None = None):
        self._env = env

        # Hyperparameters (use default if not provided)
        self.hyperparams = hyperparams if hyperparams is not None else Hyperparameters()

        # Phase controller for state machine
        self.phase_controller = create_controller(GamePhase.GATHER_GERMANIUM)

        # Action / object names
        self._action_names: List[str] = env.action_names
        self._object_type_names: List[str] = env.object_type_names

        # Lookups
        self._action_lookup: Dict[str, int] = {name: i for i, name in enumerate(self._action_names)}
        obs_features = env.observation_features
        self._feature_name_to_id: Dict[str, int] = {f.name: f.id for f in obs_features.values()}

        # Initialize move action indices
        self._MOVE_N = self._action_lookup.get("move_north", -1)
        self._MOVE_S = self._action_lookup.get("move_south", -1)
        self._MOVE_E = self._action_lookup.get("move_east", -1)
        self._MOVE_W = self._action_lookup.get("move_west", -1)
        self._MOVE_SET = {self._MOVE_N, self._MOVE_S, self._MOVE_E, self._MOVE_W}

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
            "assembler": "heart",
            "chest": "chest",
        }

        # Phase-specific glyph overrides for crafting
        self._phase_to_glyph: Dict[GamePhase, str] = {
            GamePhase.CRAFT_UNCLIP_ITEM: "gear",
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
            GamePhase.UNCLIP_STATION: None,  # Dynamic
            GamePhase.CRAFT_UNCLIP_ITEM: "assembler",
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

        # ADAPTIVE EXPLORATION: Scale based on map size
        import math

        map_size = self._map_height * self._map_width
        base_exploration = self.hyperparams.exploration_phase_steps
        scale_factor = math.sqrt(map_size / 1600.0)
        self._adaptive_exploration_steps = int(base_exploration * scale_factor)
        logger.info(
            f"[AdaptiveExploration] Map {self._map_height}x{self._map_width} ({map_size} cells), "
            f"exploration steps: {self._adaptive_exploration_steps}"
        )

        # Incremental knowledge - NO OMNISCIENCE
        self._station_positions: Dict[str, Tuple[int, int]] = {}  # discovered stations (legacy)
        self._visited_cells: set[Tuple[int, int]] = set()

        # Occupancy grid: 0=unknown, 1=free, 2=obstacle
        self._occ = [[self.OCC_UNKNOWN for _ in range(self._map_width)] for _ in range(self._map_height)]

        # Movement tracking
        self._prev_pos: Optional[Tuple[int, int]] = None
        self._last_action_idx: Optional[int] = None
        self._last_attempt_was_use: bool = False

        # Extractor memory system
        self.extractor_memory = ExtractorMemory(hyperparams=self.hyperparams)

        # Load unclip recipes from environment config
        self._unclip_recipes = self._load_unclip_recipes_from_config()

        # Navigator for pathfinding
        self.navigator = Navigator(self._map_height, self._map_width)

        # Defensive state cache - persist state even if wrapper doesn't pass it
        self._cached_state: Optional[AgentState] = None

        # Station type to resource type mapping
        self._station_to_resource_type: Dict[str, str] = {
            "carbon_extractor": "carbon",
            "oxygen_extractor": "oxygen",
            "germanium_extractor": "germanium",
            "silicon_extractor": "silicon",
            "charger": "charger",
        }

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

    @property
    def RECHARGE_START(self) -> int:
        """Dynamic recharge start threshold based on map size."""
        map_size = max(self._map_height, self._map_width)
        return self.hyperparams.recharge_start_large if map_size >= 50 else self.hyperparams.recharge_start_small

    @property
    def RECHARGE_STOP(self) -> int:
        """Dynamic recharge stop threshold based on map size."""
        map_size = max(self._map_height, self._map_width)
        return self.hyperparams.recharge_stop_large if map_size >= 50 else self.hyperparams.recharge_stop_small

    # ========== Helper Methods ==========

    def _can_reach_safely(
        self,
        target_pos: Tuple[int, int],
        state: AgentState,
        task_energy: int = 0,
        is_recharge: bool = False,
        regen_per_step: float = None,
    ) -> bool:
        """Check if agent has enough energy to reach target and complete task, with passive regen.

        - Each move costs 1 energy; regen_per_step restores energy per step.
        - If regen_per_step >= 1, movement is effectively free (or net-positive).
        """
        if state.agent_row == -1:
            return True  # Position unknown, assume OK

        # Use default regen if not specified
        if regen_per_step is None:
            regen_per_step = 1.0  # Default passive regen

        distance = abs(target_pos[0] - state.agent_row) + abs(target_pos[1] - state.agent_col)
        net_cost_per_step = 1.0 - regen_per_step  # positive means net drain

        if is_recharge:
            effective_cost = max(0.0, distance * net_cost_per_step)
            buffer = 5  # Fixed buffer
            required = effective_cost + buffer
            return state.energy >= required or net_cost_per_step <= 0

        # For gathering:
        trip_steps = distance * (2 if task_energy < 50 else 1)
        effective_cost = max(0.0, trip_steps * net_cost_per_step)
        buffer = 10 if task_energy < 50 else 5  # Fixed buffer
        required = effective_cost + task_energy + buffer

        if net_cost_per_step <= 0:
            return True  # movement free / net-positive
        return state.energy >= required

    def _exists_viable_alternative(self, resource_type: str, state: AgentState, radius: int) -> bool:
        """Check if there's a viable alternative extractor within radius."""
        current_pos = (state.agent_row, state.agent_col)
        extractors = self.extractor_memory.get_by_type(resource_type)

        for e in extractors:
            if e.position == state.wait_target:
                continue  # Skip current target
            dist = abs(e.position[0] - current_pos[0]) + abs(e.position[1] - current_pos[1])
            if dist <= radius and not e.is_low(0.25):  # Fixed depletion threshold
                rem = self.cooldown_remaining(e, state.step_count)
                if rem < 3:  # Fixed rotation threshold
                    return True
        return False

    def _navigate_to_best_alternative(self, resource_type: str, state: AgentState) -> int:
        """Navigate to the best alternative extractor."""
        current_pos = (state.agent_row, state.agent_col)
        extractors = self.extractor_memory.get_by_type(resource_type)

        # Filter viable alternatives
        alternatives = []
        for e in extractors:
            if e.position == state.wait_target:
                continue  # Skip current target
            if not e.is_low(0.25):  # Fixed depletion threshold
                rem = self.cooldown_remaining(e, state.step_count)
                if rem < 3:  # Fixed rotation threshold
                    alternatives.append(e)

        if not alternatives:
            # No alternatives, explore
            return self._explore_simple(state)

        # Find best alternative
        best = self._find_best_extractor(alternatives, current_pos, resource_type)
        if not best:
            return self._explore_simple(state)

        # Navigate to best alternative
        target_pos = best.position
        start_pos = (state.agent_row, state.agent_col)
        result = self.navigator.navigate_to(
            start=start_pos,
            target=target_pos,
            occupancy_map=self._occ,
            optimistic=True,  # Fixed optimistic planning
            use_astar=True,  # Fixed A* usage
            astar_threshold=20,  # Fixed A* threshold
        )

        if result.next_step:
            nr, nc = result.next_step
            dr, dc = nr - state.agent_row, nc - state.agent_col
            self._last_attempt_was_use = False
            logger.debug(f"[Rotate] Moving toward alternative {resource_type} at {target_pos}")
            return self._step_toward(dr, dc)

        return self._explore_simple(state)

    def _update_extractor_after_use(
        self, pos: Tuple[int, int], state: AgentState, resource_gained: int, resource_type: str
    ):
        """Update extractor info after using it."""
        extractor = self.extractor_memory.get_at_position(pos)
        if not extractor:
            return

        extractor.update_after_use(resource_gained, state.step_count)
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
        if not hasattr(self, "_prev_inventory"):
            self._prev_inventory = current.copy()
            return changes

        for resource, amount in current.items():
            prev = self._prev_inventory.get(resource, amount)
            if amount != prev:
                changes[resource] = amount - prev

        # Update for next time
        self._prev_inventory = current.copy()
        return changes

    def cooldown_remaining(self, extractor: ExtractorInfo, current_step: int) -> int:
        """Get remaining cooldown turns for an extractor (observed or estimated).

        NOTE: observed_cooldown_remaining can be stale if the extractor goes out of view.
        We prioritize estimation based on last_used_step when available.
        """
        # Prefer estimation based on last_used_step if available
        if extractor.last_used_step >= 0:
            total = extractor.learned_cooldown if extractor.learned_cooldown is not None else 20
            elapsed = max(0, current_step - extractor.last_used_step)
            return max(0, total - elapsed)

        # If never used by us, assume ready (ignore stale observed_cooldown)
        return 0

    def _find_best_extractor_for_phase(self, phase: GamePhase, state: AgentState) -> Optional[Tuple[int, int]]:
        """Find best extractor for current gathering phase using extractor memory.

        Returns position of best extractor, or None if should explore (or wait).
        """
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

        task_energy = 50 if resource_type == "silicon" else 0
        all_extractors = self.extractor_memory.get_by_type(resource_type)
        logger.info(f"[Phase1] Finding {resource_type}: {len(all_extractors)} total discovered")

        # Find best available extractor
        current_pos = (state.agent_row, state.agent_col)
        best = self.extractor_memory.find_best_extractor(
            resource_type, current_pos, state.step_count, self.cooldown_remaining
        )

        if best is None:
            # No available extractors - consider waiting for the shortest cooldown if it's short and close
            if len(all_extractors) > 0:
                extractors_on_cooldown = [e for e in all_extractors if not e.is_depleted()]
                if extractors_on_cooldown:

                    def est(e: ExtractorInfo) -> int:
                        return self.cooldown_remaining(e, state.step_count)

                    cand = min(extractors_on_cooldown, key=est)
                    cooldown_time = est(cand)
                    dist_to_extractor = abs(cand.position[0] - state.agent_row) + abs(
                        cand.position[1] - state.agent_col
                    )
                    should_wait = (
                        cooldown_time < 10
                        or (cooldown_time < 20 and dist_to_extractor < 5)
                        or (cooldown_time < 100 and dist_to_extractor <= 1)
                    )
                    if should_wait:
                        logger.info(
                            f"[Phase3] Waiting for {resource_type} at {cand.position} "
                            f"(cooldown~{cooldown_time}, dist={dist_to_extractor})"
                        )
                        state.wait_target = cand.position
                        if state.waiting_since_step < 0:
                            state.waiting_since_step = state.step_count
                        return cand.position

            logger.info(
                f"[Phase1] No available {resource_type} extractors "
                f"(found {len(all_extractors)} but all depleted/on cooldown)"
            )
            return None

        distance = abs(best.position[0] - state.agent_row) + abs(best.position[1] - state.agent_col)
        logger.info(f"[Phase1] Best {resource_type} extractor: {best.position}, distance={distance}")

        # Check energy feasibility
        is_charger = resource_type == "charger"
        if not self._can_reach_safely(best.position, state, task_energy, is_recharge=is_charger):
            logger.warning(
                f"[Phase1] Not enough energy to reach {resource_type} at {best.position} (have {state.energy})"
            )
            return None

        logger.info(f"[Phase1] ✓ Selected {resource_type} at {best.position}")
        return best.position

    # ---------- Policy Interface ----------
    def agent_state(self) -> AgentState:
        return AgentState()

    def _ensure_move_indices(self) -> None:
        """Defensive: ensure move indices are initialized (should already be done in __init__)."""
        if hasattr(self, "_MOVE_SET"):
            return
        self._MOVE_N = self._action_lookup.get("move_north", -1)
        self._MOVE_S = self._action_lookup.get("move_south", -1)
        self._MOVE_E = self._action_lookup.get("move_east", -1)
        self._MOVE_W = self._action_lookup.get("move_west", -1)
        self._MOVE_SET = {a for a in (self._MOVE_N, self._MOVE_S, self._MOVE_E, self._MOVE_W) if a != -1}

    def step_with_state(
        self, obs: MettaGridObservation, state: Optional[AgentState]
    ) -> tuple[MettaGridAction, Optional[AgentState]]:
        """Main policy step: update knowledge, select phase, choose an action."""
        # Defensive: use cached state if provided state is None or fresh
        if state is None or (state.step_count == 0 and self._cached_state is not None):
            state = self._cached_state if self._cached_state is not None else self.agent_state()

        state.step_count += 1

        # Cache the state for next call
        self._cached_state = state

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
        self._last_obs = obs  # Store for inventory checks in phase determination
        self._discover_stations_from_observation(obs, state)
        self._update_wall_knowledge(state)

        # === Track extractor usage ===
        inv_changes = self._detect_inventory_changes(state)

        # If we gained resources adjacent to a known extractor, update its stats
        if inv_changes and state.agent_row >= 0:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                check_pos = (state.agent_row + dr, state.agent_col + dc)
                extractor = self.extractor_memory.get_at_position(check_pos)
                if extractor:
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
            state.hearts_assembled += 1
            state.wait_counter = 0
            state.current_phase = GamePhase.GATHER_GERMANIUM
            state.just_deposited = True
            logger.info(f"Step {state.step_count}: Heart deposited! Total hearts: {state.hearts_assembled}")

        # Decide phase & act
        old_phase = state.current_phase
        state.current_phase = self._determine_phase(state, obs)

        # Track phase transitions for oscillation detection
        if state.current_phase != old_phase:
            state.phase_entry_step = state.step_count
            state.phase_entry_inventory = {
                "germanium": state.germanium,
                "silicon": state.silicon,
                "carbon": state.carbon,
                "oxygen": state.oxygen,
                "decoder": state.decoder,
            }
            phase_name = state.current_phase.name
            if phase_name.startswith("GATHER_"):
                resource_name = phase_name.replace("GATHER_", "").lower()
                if state.phase_visit_count is not None:
                    state.phase_visit_count[resource_name] = state.phase_visit_count.get(resource_name, 0) + 1
                current_amount = getattr(state, resource_name)
                if (
                    state.resource_progress_tracking is not None
                    and resource_name not in state.resource_progress_tracking
                ):
                    state.resource_progress_tracking[resource_name] = current_amount
                if state.resource_gathering_start is not None:
                    state.resource_gathering_start[resource_name] = state.step_count

        action_idx = self._execute_phase(state)

        # Bookkeeping
        self._last_action_idx = action_idx
        self._prev_pos = (state.agent_row, state.agent_col)
        state.last_heart = state.heart

        return dtype_actions.type(action_idx), state

    # ---------- Observation → Knowledge ----------
    def _discover_stations_from_observation(self, obs: MettaGridObservation, state: AgentState) -> None:
        """Discover stations/walls from vision and update occupancy grid.

        Also reads cooldown_remaining and remaining_uses from observations to update extractor state.
        """
        if state.agent_row == -1:
            return

        type_id_feature = self._feature_name_to_id.get("type_id", 0)
        converting_feature = self._feature_name_to_id.get("converting", 5)
        cooldown_feature = self._feature_name_to_id.get("cooldown_remaining", 14)
        clipped_feature = self._feature_name_to_id.get("clipped", 15)
        remaining_uses_feature = self._feature_name_to_id.get("remaining_uses", 16)

        # First pass: collect all features by position
        position_features = {}

        for tok in obs:
            feature_id = self._to_int(tok[1])

            # Decode local coords (hi nibble=row, lo nibble=col)
            packed = self._to_int(tok[0])
            obs_r = packed >> 4
            obs_c = packed & 0x0F

            # Convert to absolute map coords
            map_r = obs_r - self.OBS_HEIGHT_RADIUS + state.agent_row
            map_c = obs_c - self.OBS_WIDTH_RADIUS + state.agent_col
            if not self._is_valid_position(map_r, map_c):
                continue

            pos = (map_r, map_c)
            if pos not in position_features:
                position_features[pos] = {}

            if feature_id == type_id_feature:
                position_features[pos]["type_id"] = self._to_int(tok[2])
            elif feature_id == converting_feature:
                position_features[pos]["converting"] = self._to_int(tok[2])
            elif feature_id == cooldown_feature:
                position_features[pos]["cooldown_remaining"] = self._to_int(tok[2])
            elif feature_id == clipped_feature:
                position_features[pos]["clipped"] = self._to_int(tok[2])
            elif feature_id == remaining_uses_feature:
                position_features[pos]["remaining_uses"] = self._to_int(tok[2])

        # Second pass: process positions with their features
        for pos, features in position_features.items():
            map_r, map_c = pos
            type_id = features.get("type_id")
            if type_id is None:
                continue

            if type_id == self._wall_type_id:
                # Wall (unwalkable)
                self._mark_cell(map_r, map_c, self.OCC_OBSTACLE)
                continue

            if type_id in self._type_id_to_station:
                # Station (unwalkable). Remember first seen location.
                station_name = self._type_id_to_station[type_id]
                self._mark_cell(map_r, map_c, self.OCC_OBSTACLE)

                if station_name not in self._station_positions:
                    self._station_positions[station_name] = pos
                    logger.info(f"Discovered {station_name} at {pos}")

                    # Track critical station discovery
                    if station_name == "assembler" and not state.assembler_discovered:
                        state.assembler_discovered = True
                        logger.info(f"[Phase3] ✓ Assembler discovered at {pos}")
                    elif station_name == "chest" and not state.chest_discovered:
                        state.chest_discovered = True
                        logger.info(f"[Phase3] ✓ Chest discovered at {pos}")

                # Add to extractor memory if it's a resource extractor
                resource_type = self._station_to_resource_type.get(station_name)
                if resource_type:
                    extractor = self.extractor_memory.add_extractor(pos, resource_type, station_name)

                    # Initialize reasonable cooldown defaults if unknown
                    if extractor.learned_cooldown is None:
                        default_cooldowns = {
                            "germanium": 0,
                            "silicon": 0,
                            "carbon": 10,
                            "oxygen": 100,
                            "charger": 10,
                        }
                        extractor.learned_cooldown = default_cooldowns.get(resource_type, 10)

                    # Observations
                    if "converting" in features:
                        extractor.observed_converting = bool(features["converting"])

                    # ---- CLIPPED STATUS INTERPRETATION ----
                    if "clipped" in features:
                        was_clipped = extractor.is_clipped
                        extractor.is_clipped = bool(features["clipped"])
                        if extractor.is_clipped and not was_clipped:
                            logger.info(f"[ObsUpdate] {resource_type} at {pos} is CLIPPED")
                        elif not extractor.is_clipped and was_clipped:
                            logger.info(f"[ObsUpdate] {resource_type} at {pos} was UNCLIPPED successfully!")
                            if state.unclip_target == pos:
                                state.unclip_target = None
                                logger.info(f"[Unclip] Cleared unclip target (obs shows unclipped) for {pos}")
                    else:
                        # No 'clipped' feature observed this step ⇒ treat as UNCLIPPED
                        if extractor.is_clipped:
                            extractor.is_clipped = False
                            logger.info(
                                f"[ObsUpdate] {resource_type} at {pos} UNCLIPPED (implicit: no 'clipped' token present)"
                            )
                            if state.unclip_target == pos:
                                state.unclip_target = None
                                logger.info(f"[Unclip] Cleared unclip target (implicit unclipped) for {pos}")

                    if "cooldown_remaining" in features:
                        cv = features["cooldown_remaining"]
                        extractor.observed_cooldown_remaining = cv
                        # Learn cooldown from observation: elapsed + remaining = total cooldown
                        if extractor.last_used_step >= 0:
                            elapsed = state.step_count - extractor.last_used_step
                            if cv > 0:
                                total = elapsed + cv
                                if extractor.learned_cooldown is None or elapsed < 5:
                                    extractor.learned_cooldown = total
                            elif cv == 0 and elapsed > 0:
                                if extractor.learned_cooldown is None or elapsed < 20:
                                    extractor.learned_cooldown = elapsed

                    if "remaining_uses" in features:
                        uses_val = features["remaining_uses"]
                        extractor.uses_remaining_fraction = min(1.0, uses_val / 50.0)
                        if uses_val == 0 and not extractor.permanently_depleted:
                            extractor.permanently_depleted = True
                            logger.warning(
                                f"[ObsUpdate] {resource_type} at {pos} has 0 uses remaining - marking DEPLETED"
                            )
                continue

            # Non-agent, non-wall, non-station object: treat as free
            if not self._object_type_names[type_id].startswith("agent"):
                self._mark_cell(map_r, map_c, self.OCC_FREE)

        # Ensure current cell is free
        self._mark_cell(state.agent_row, state.agent_col, self.OCC_FREE)

    def _update_wall_knowledge(self, state: AgentState) -> None:
        """Update occupancy based on movement results.

        - If we moved successfully, mark new cell as free
        - If we tried to move but didn't, mark intended target as obstacle
        - UNLESS we were trying to USE a station (staying is expected)
        """
        self._ensure_move_indices()
        if not self._prev_pos or self._last_action_idx not in self._MOVE_SET:
            return

        cur = (state.agent_row, state.agent_col)
        if cur != self._prev_pos:
            # Moved: mark new cell free
            self._mark_cell(cur[0], cur[1], self.OCC_FREE)
            return

        # If we were trying to USE a station, staying in place is expected - don't mark as obstacle
        if self._last_attempt_was_use:
            logger.info("[WallKnowledge] Stayed in place while using station - expected (resetting flag)")
            self._last_attempt_was_use = False  # Reset for next action
            return

        # Didn't move: intended target is blocked
        dr, dc = self._action_to_dir(self._last_action_idx)
        if dr is None or dc is None:
            return

        wr, wc = self._prev_pos[0] + dr, self._prev_pos[1] + dc
        if not self._is_valid_position(wr, wc):
            return

        # Mark as obstacle (could be a wall or a station; usable via adjacency)
        if self._occ[wr][wc] != self.OCC_OBSTACLE:
            logger.info(f"Marking blocked cell at ({wr},{wc})")
        self._occ[wr][wc] = self.OCC_OBSTACLE

    # ---------- Phase & Action Selection ----------
    def _determine_phase(self, state: AgentState, obs: MettaGridObservation) -> GamePhase:
        """Choose the current high-level phase using the phase controller."""
        # Create context for phase controller
        ctx = Context(obs=obs, env=self._env, step=state.step_count)
        # Add policy implementation to context for guards
        ctx.policy_impl = self

        # Update phase using the controller
        new_phase = self.phase_controller.maybe_transition(state, ctx, logger)

        # Update state's current phase
        state.current_phase = new_phase

        return new_phase

    def _determine_phase_greedy(self, state: AgentState, germ_needed: int) -> GamePhase:
        """GREEDY_OPPORTUNISTIC: Always grab closest needed resource."""
        has_all = (
            state.germanium >= germ_needed
            and state.silicon >= self.SILICON_REQ
            and state.carbon >= self.CARBON_REQ
            and state.oxygen >= self.OXYGEN_REQ
        )

        if has_all and state.energy >= self.ENERGY_REQ:
            assembler_pos = self._station_positions.get("assembler")
            if assembler_pos:
                return GamePhase.ASSEMBLE_HEART

        needed = []
        if state.germanium < germ_needed:
            needed.append(("germanium", GamePhase.GATHER_GERMANIUM))
        if state.silicon < self.SILICON_REQ and state.energy >= self.hyperparams.min_energy_for_silicon:
            needed.append(("silicon", GamePhase.GATHER_SILICON))
        if state.carbon < self.CARBON_REQ:
            needed.append(("carbon", GamePhase.GATHER_CARBON))
        if state.oxygen < self.OXYGEN_REQ:
            needed.append(("oxygen", GamePhase.GATHER_OXYGEN))

        if not needed:
            return GamePhase.EXPLORE

        # Pick the needed phase with the closest available extractor
        best_phase = None
        best_dist = float("inf")
        for resource_type, phase in needed:
            for e in self.extractor_memory.get_by_type(resource_type):
                if e.is_depleted():
                    continue
                if not e.is_available(state.step_count, self.cooldown_remaining):
                    continue
                dist = abs(e.position[0] - state.agent_row) + abs(e.position[1] - state.agent_col)
                if dist < best_dist:
                    best_dist = dist
                    best_phase = phase

        return best_phase if best_phase else needed[0][1]

    def _determine_phase_explorer_first(self, state: AgentState, germ_needed: int) -> GamePhase:
        """EXPLORER_FIRST: Explore for N steps (adaptive to map size), then gather greedily."""
        if state.step_count < self._adaptive_exploration_steps:
            return GamePhase.EXPLORE
        return self._determine_phase_greedy(state, germ_needed)

    def _determine_phase_sequential(self, state: AgentState, germ_needed: int) -> GamePhase:
        """SEQUENTIAL_SIMPLE: Fixed order G→Si→C→O."""
        has_all = (
            state.germanium >= germ_needed
            and state.silicon >= self.SILICON_REQ
            and state.carbon >= self.CARBON_REQ
            and state.oxygen >= self.OXYGEN_REQ
        )
        if has_all and state.energy >= self.ENERGY_REQ:
            assembler_pos = self._station_positions.get("assembler")
            if assembler_pos:
                return GamePhase.ASSEMBLE_HEART

        if state.germanium < germ_needed:
            return GamePhase.GATHER_GERMANIUM
        if state.silicon < self.SILICON_REQ and state.energy >= self.hyperparams.min_energy_for_silicon:
            return GamePhase.GATHER_SILICON
        if state.carbon < self.CARBON_REQ:
            return GamePhase.GATHER_CARBON
        if state.oxygen < self.OXYGEN_REQ:
            return GamePhase.GATHER_OXYGEN

        return GamePhase.EXPLORE

    def _determine_phase_efficiency_learner(self, state: AgentState, germ_needed: int) -> GamePhase:
        """EFFICIENCY_LEARNER: Learn extractor efficiency, prioritize best ones."""
        has_all = (
            state.germanium >= germ_needed
            and state.silicon >= self.SILICON_REQ
            and state.carbon >= self.CARBON_REQ
            and state.oxygen >= self.OXYGEN_REQ
        )
        if has_all and state.energy >= self.ENERGY_REQ:
            assembler_pos = self._station_positions.get("assembler")
            if assembler_pos:
                return GamePhase.ASSEMBLE_HEART

        needed = []
        if state.germanium < germ_needed:
            needed.append(("germanium", GamePhase.GATHER_GERMANIUM))
        if state.silicon < self.SILICON_REQ and state.energy >= self.hyperparams.min_energy_for_silicon:
            needed.append(("silicon", GamePhase.GATHER_SILICON))
        if state.carbon < self.CARBON_REQ:
            needed.append(("carbon", GamePhase.GATHER_CARBON))
        if state.oxygen < self.OXYGEN_REQ:
            needed.append(("oxygen", GamePhase.GATHER_OXYGEN))

        if not needed:
            return GamePhase.EXPLORE

        best_phase = None
        best_score = -float("inf")
        for resource_type, phase in needed:
            for e in self.extractor_memory.get_by_type(resource_type):
                if e.is_depleted():
                    continue
                if not e.is_available(state.step_count, self.cooldown_remaining):
                    continue
                dist = abs(e.position[0] - state.agent_row) + abs(e.position[1] - state.agent_col)
                efficiency = e.avg_output() if e.avg_output() > 0 else 5
                score = efficiency * 10 - dist
                if score > best_score:
                    best_score = score
                    best_phase = phase

        return best_phase if best_phase else needed[0][1]

    # ========== END STRATEGY METHODS ==========

    # Old helper methods removed - phase controller handles transitions

    def _execute_phase(self, state: AgentState) -> int:
        """Convert phase to a concrete action (move or glyph change)."""
        # EXPLORE phase
        if state.current_phase == GamePhase.EXPLORE:
            self._last_attempt_was_use = False
            plan = self._plan_to_frontier_action(state)
            return plan if plan is not None else self._explore_simple(state)

        # UNCLIP_STATION phase - unclip extractor with decoder (GENERALIZED)
        if state.current_phase == GamePhase.UNCLIP_STATION:
            # Prefer an existing target if still clipped; else choose nearest clipped extractor of ANY type.
            target_pos = None

            # If we already had a target, keep it if still clipped (based on latest obs)
            if state.unclip_target is not None:
                cur = self.extractor_memory.get_at_position(state.unclip_target)
                if cur and cur.is_clipped:
                    target_pos = state.unclip_target
                else:
                    # Target no longer clipped or unknown => clear
                    logger.info(f"[Unclip] Previous target {state.unclip_target} no longer clipped; clearing.")
                    state.unclip_target = None

            if target_pos is None:
                # Find nearest clipped extractor across all resource types
                clipped: List[ExtractorInfo] = [e for e in self.extractor_memory.get_all() if e.is_clipped]
                if clipped:
                    # Choose nearest by Manhattan distance
                    def md(e: ExtractorInfo) -> int:
                        return abs(e.position[0] - state.agent_row) + abs(e.position[1] - state.agent_col)

                    chosen = min(clipped, key=md)
                    target_pos = chosen.position
                    state.unclip_target = target_pos
                    logger.info(f"[Unclip] New unclip target set to {target_pos} ({chosen.resource_type})")
                else:
                    logger.warning("[Unclip] No clipped extractors known; exploring.")
                    self._last_attempt_was_use = False
                    plan = self._plan_to_frontier_action(state)
                    return plan if plan is not None else self._explore_simple(state)

            start_pos = (state.agent_row, state.agent_col)
            result = self.navigator.navigate_to(
                start=start_pos,
                target=target_pos,
                occupancy_map=self._occ,
                optimistic=True,
                use_astar=True,
                astar_threshold=20,
            )

            # Adjacent - attempt to unclip by walking into the extractor
            if result.is_adjacent:
                tr, tc = target_pos
                dr, dc = tr - state.agent_row, tc - state.agent_col
                self._last_attempt_was_use = True
                logger.info(f"[Unclip] Using decoder to unclip at {target_pos} (decoder={state.decoder})")
                return self._step_toward(dr, dc)

            # Navigator found a next step
            if result.next_step:
                nr, nc = result.next_step
                dr, dc = nr - state.agent_row, nc - state.agent_col
                self._last_attempt_was_use = False
                logger.debug(f"[Unclip] Moving toward {target_pos}")
                return self._step_toward(dr, dc)

            # Can't reach - explore
            logger.warning(f"[Unclip] Cannot reach extractor at {target_pos}, exploring")
            self._last_attempt_was_use = False
            plan = self._plan_to_frontier_action(state)
            return plan if plan is not None else self._explore_simple(state)

        # CRAFT_UNCLIP_ITEM phase
        if state.current_phase == GamePhase.CRAFT_UNCLIP_ITEM:
            # Check if we need to switch to gear glyph first
            needed_glyph = self._phase_to_glyph.get(state.current_phase, "gear")
            if state.current_glyph != needed_glyph:
                state.current_glyph = needed_glyph
                state.wait_counter = 0
                glyph_id = self._glyph_name_to_id.get(needed_glyph, 0)
                logger.info(f"[Craft] Switching to {needed_glyph} glyph (ID: {glyph_id})")
                return self._action_lookup.get(f"change_glyph_{glyph_id}", self._action_lookup.get("noop", 0))

            assembler_pos = self._station_positions.get("assembler")
            if not assembler_pos:
                logger.warning("[Craft] No assembler found, exploring")
                self._last_attempt_was_use = False
                plan = self._plan_to_frontier_action(state)
                return plan if plan is not None else self._explore_simple(state)

            start_pos = (state.agent_row, state.agent_col)
            result = self.navigator.navigate_to(
                start=start_pos,
                target=assembler_pos,
                occupancy_map=self._occ,
                optimistic=True,
                use_astar=True,
                astar_threshold=20,
            )

            # Adjacent - ready to craft by walking into the assembler
            if result.is_adjacent:
                tr, tc = assembler_pos
                dr, dc = tr - state.agent_row, tc - state.agent_col
                self._last_attempt_was_use = True
                action = self._step_toward(dr, dc)
                action_name = self._action_names[action] if action < len(self._action_names) else f"action_{action}"
                logger.info(
                    f"[Craft] Adjacent to assembler at {assembler_pos}, USE via {action_name} (carbon={state.carbon})"
                )
                return action

            # Navigator found a next step
            if result.next_step:
                nr, nc = result.next_step
                dr, dc = nr - state.agent_row, nc - state.agent_col
                self._last_attempt_was_use = False
                logger.debug(f"[Craft] Moving toward assembler at {assembler_pos}")
                return self._step_toward(dr, dc)

            # Can't reach - explore
            logger.warning(f"[Craft] Cannot reach assembler at {assembler_pos}, exploring")
            self._last_attempt_was_use = False
            plan = self._plan_to_frontier_action(state)
            return plan if plan is not None else self._explore_simple(state)

        # Continue with other phases...
        station = self._phase_to_station.get(state.current_phase)
        if not station:
            return self._action_lookup.get("noop", 0)

        # Glyph switching - check for phase-specific glyph first
        needed_glyph = self._phase_to_glyph.get(state.current_phase, self._station_to_glyph.get(station, "default"))
        if state.current_glyph != needed_glyph:
            state.current_glyph = needed_glyph
            state.wait_counter = 0
            glyph_id = self._glyph_name_to_id.get(needed_glyph, 0)
            return self._action_lookup.get(f"change_glyph_{glyph_id}", self._action_lookup.get("noop", 0))

        # Gathering phases use extractor memory
        is_gathering_phase = state.current_phase in [
            GamePhase.GATHER_CARBON,
            GamePhase.GATHER_OXYGEN,
            GamePhase.GATHER_GERMANIUM,
            GamePhase.GATHER_SILICON,
            GamePhase.RECHARGE,
        ]

        target_pos = None
        if is_gathering_phase and state.agent_row != -1:
            target_pos = self._find_best_extractor_for_phase(state.current_phase, state)
            if target_pos is None:
                # No available extractors - explore to find more
                self._last_attempt_was_use = False
                logger.info(f"[Phase1] {state.current_phase.value}: No available extractors, exploring")
                plan = self._plan_to_frontier_action(state)
                return plan if plan is not None else self._explore_simple(state)
            else:
                logger.info(f"[Phase1] {state.current_phase.value}: Targeting extractor at {target_pos}")

        # Fallback to legacy single-station logic
        if target_pos is None:
            if station in self._station_positions and state.agent_row != -1:
                target_pos = self._station_positions[station]
            else:
                self._last_attempt_was_use = False
                plan = self._plan_to_frontier_action(state)
                return plan if plan is not None else self._explore_simple(state)

        # Navigate
        start_pos = (state.agent_row, state.agent_col)
        dist = abs(target_pos[0] - start_pos[0]) + abs(target_pos[1] - start_pos[1])

        result = self.navigator.navigate_to(
            start=start_pos,
            target=target_pos,
            occupancy_map=self._occ,
            optimistic=True,
            use_astar=True,
            astar_threshold=20,
        )

        # Adjacent - ready to "use" by walking into the station.
        if result.is_adjacent:
            e = self.extractor_memory.get_at_position(target_pos)
            if e and state.wait_target == target_pos:
                rem = self.cooldown_remaining(e, state.step_count)
                if rem > self.hyperparams.wait_if_cooldown_leq:
                    # Check if we should rotate to an alternative
                    if self._exists_viable_alternative(e.resource_type, state, 7):  # Fixed radius
                        self._last_attempt_was_use = False
                        logger.debug(f"[Wait] At {target_pos}, rotating to alternative (cooldown~{rem})")
                        return self._navigate_to_best_alternative(e.resource_type, state)

                    # Otherwise wait but cap patience
                    if state.waiting_since_step < 0:
                        state.waiting_since_step = state.step_count

                    if state.step_count - state.waiting_since_step <= 12:  # Fixed patience
                        self._last_attempt_was_use = False
                        logger.debug(f"[Wait] At {target_pos}, remaining cooldown~{rem}; idling (noop).")
                        return self._action_lookup.get("noop", 0)
                    else:
                        # Patience exhausted: force rotate
                        logger.debug(f"[Wait] At {target_pos}, patience exhausted, rotating")
                        return self._navigate_to_best_alternative(e.resource_type, state)
                else:
                    # Try-use when rem <= wait_if_cooldown_leq
                    state.wait_target = None  # Clear wait target

            tr, tc = target_pos
            dr, dc = tr - state.agent_row, tc - state.agent_col
            self._last_attempt_was_use = True
            action = self._step_toward(dr, dc)
            action_name = self._action_names[action] if action < len(self._action_names) else f"action_{action}"
            logger.info(
                f"[Navigation] Adjacent to {target_pos}, USE via {action_name} "
                f"(G={state.germanium} Si={state.silicon} C={state.carbon} O={state.oxygen} E={state.energy})"
            )
            return action

        # Navigator found a next step
        if result.next_step:
            nr, nc = result.next_step
            dr, dc = nr - state.agent_row, nc - state.agent_col
            self._last_attempt_was_use = False
            logger.debug(
                f"[Navigation] Moving toward {target_pos} (dist={dist}) via {result.method}: step to {result.next_step}"
            )
            return self._step_toward(dr, dc)

        # Completely stuck - explore to reveal more map
        logger.warning(f"[Navigation] No path to {target_pos} (dist={dist}, method={result.method}), exploring")
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
        """Pick a frontier by combined BFS distance and center bias when needed."""
        if state.agent_row < 0:
            return None

        start = (state.agent_row, state.agent_col)
        fronts = set(self._compute_frontiers())
        if not fronts:
            return None

        # Filter by exploration radius around home base if known; else keep all
        EXPL_RADIUS = 50
        if EXPL_RADIUS > 0:
            center = (state.home_base_row, state.home_base_col) if state.home_base_row >= 0 else start
            filtered = {(r, c) for (r, c) in fronts if abs(r - center[0]) + abs(c - center[1]) <= EXPL_RADIUS}
            if filtered:
                fronts = filtered

        # If critical stations not found, try spawn-area bias then center bias
        if not state.assembler_discovered or not state.chest_discovered:
            if state.home_base_row >= 0 and state.home_base_col >= 0:
                spawn_radius = 30
                spawn_area_frontiers = [
                    (fr, fc)
                    for fr, fc in fronts
                    if abs(fr - state.home_base_row) + abs(fc - state.home_base_col) <= spawn_radius
                ]
                if spawn_area_frontiers:
                    closest = min(spawn_area_frontiers, key=lambda p: abs(p[0] - start[0]) + abs(p[1] - start[1]))
                    logger.debug(f"[SpawnSearch] Prioritizing spawn-area frontier: {closest}")
                    return closest

            # Center bias
            map_center = (self._map_height // 2, self._map_width // 2)
            q = deque([(start, 0)])
            seen = {start}
            reachable_frontiers = []
            while q:
                (r, c), dist = q.popleft()
                for nr, nc in self._neighbors4(r, c):
                    if (nr, nc) in fronts:
                        reachable_frontiers.append(((nr, nc), dist + 1))
                for nr, nc in self._neighbors4(r, c):
                    if (nr, nc) in seen or self._occ[nr][nc] != self.OCC_FREE:
                        continue
                    seen.add((nr, nc))
                    q.append(((nr, nc), dist + 1))

            best_frontier = None
            best_score = float("inf")
            denom = max(self._map_height, self._map_width)
            for (fr, fc), bfs_dist in reachable_frontiers:
                center_dist = abs(fr - map_center[0]) + abs(fc - map_center[1])
                score = 0.5 * (bfs_dist / denom) + 0.5 * (center_dist / denom)
                if score < best_score:
                    best_score = score
                    best_frontier = (fr, fc)
            if best_frontier:
                logger.debug(f"[Phase3] Center-biased frontier: {best_frontier} (score={best_score:.2f})")
                return best_frontier

        # Default: nearest by BFS
        return self._choose_frontier_bfs(start, fronts)

    def _choose_frontier_bfs(self, start: Tuple[int, int], fronts: Set[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Pick nearest frontier by BFS distance (systematic exploration)."""
        q = deque([start])
        seen = {start}
        while q:
            r, c = q.popleft()
            for nr, nc in self._neighbors4(r, c):
                if (nr, nc) in fronts:
                    return (nr, nc)
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
        """BFS treating unknown cells as walkable; avoids only known obstacles."""
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
        step: Tuple[int, int] = goal
        while parent.get(step) is not None and parent[step] != start:
            step = parent[step]  # type: ignore[assignment]
        return step

    def _is_cell_passable(self, r: int, c: int, optimistic: bool = False) -> bool:
        """Check if a cell is passable for BFS or exploration."""
        cell_state = self._occ[r][c]
        return (cell_state != self.OCC_OBSTACLE) if optimistic else (cell_state == self.OCC_FREE)

    def _action_to_dir(self, action_idx: int) -> Tuple[Optional[int], Optional[int]]:
        """Convert action index to direction delta (dr, dc)."""
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
        """Convert direction delta to move action."""
        self._ensure_move_indices()
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
        """Heart present only if FIRST 'inv:heart' token's FIRST FIELD == 0x55."""
        heart_fid = self._feature_name_to_id.get(self.HEART_FEATURE_NAME)
        if heart_fid is None:
            return False
        for tok in obs:
            if self._to_int(tok[1]) == heart_fid:
                return self._to_int(tok[0]) == self.HEART_SENTINEL_FIRST_FIELD
        return False

    def _read_int_feature(self, obs: MettaGridObservation, feat_name: str) -> int:
        fid = self._feature_name_to_id.get(feat_name)
        if fid is None:
            return 0
        for tok in obs:
            if self._to_int(tok[1]) == fid:
                return self._to_int(tok[2])
        return 0

    def _load_unclip_recipes_from_config(self) -> Dict[str, str]:
        """
        Load unclip item recipes from environment config.
        Returns a mapping of unclip_item -> craft_resource.
        E.g., {"decoder": "carbon", "modulator": "oxygen"}
        """
        recipes = {}

        # Try to access assembler config from environment
        try:
            # Access the config through the env_cfg attribute
            env_cfg = getattr(self._env, 'env_cfg', None)
            if env_cfg and hasattr(env_cfg, 'game') and hasattr(env_cfg.game, 'objects'):
                assembler_config = env_cfg.game.objects.get("assembler")
                if assembler_config and hasattr(assembler_config, "recipes"):
                    for glyph_seq, protocol in assembler_config.recipes:
                        # Look for gear glyph recipes
                        if "gear" in glyph_seq:
                            # Get the output (unclip item)
                            output_resources = protocol.output_resources
                            input_resources = protocol.input_resources

                            # Map unclip item to the resource needed to craft it
                            for unclip_item in output_resources:
                                if unclip_item in ["decoder", "modulator", "resonator", "scrambler"]:
                                    # Find the input resource (should be only one)
                                    for craft_resource in input_resources:
                                        if craft_resource in ["carbon", "oxygen", "germanium", "silicon"]:
                                            recipes[unclip_item] = craft_resource
                                            logger.info(f"[Recipes] {unclip_item} requires {craft_resource}")
                                            break
        except Exception as e:
            logger.warning(f"[Recipes] Could not load unclip recipes from config: {e}")

        # Fallback to default mappings if config reading fails
        if not recipes:
            recipes = {
                "decoder": "carbon",
                "modulator": "oxygen",
                "resonator": "silicon",
                "scrambler": "germanium",
            }
            logger.info("[Recipes] Using default unclip recipe mappings")

        return recipes

    def _update_inventory(self, obs: MettaGridObservation, state: AgentState) -> None:
        """Read inventory strictly from observation tokens."""
        state.carbon = self._read_int_feature(obs, "inv:carbon")
        state.oxygen = self._read_int_feature(obs, "inv:oxygen")
        state.germanium = self._read_int_feature(obs, "inv:germanium")
        state.silicon = self._read_int_feature(obs, "inv:silicon")
        state.energy = self._read_int_feature(obs, "inv:energy")
        state.heart = 1 if self._has_heart_from_obs(obs) else 0
        # Unclip items
        state.decoder = self._read_int_feature(obs, "inv:decoder")
        state.modulator = self._read_int_feature(obs, "inv:modulator")
        state.resonator = self._read_int_feature(obs, "inv:resonator")
        state.scrambler = self._read_int_feature(obs, "inv:scrambler")

        # Update state's recipe knowledge
        if not state.unclip_recipes:
            state.unclip_recipes = self._unclip_recipes

    # ---------- Exploration fallback ----------
    def _explore_simple(self, state: AgentState) -> int:
        """Boustrophedon sweep with simple preference for not-recent cells."""
        if state.agent_row == -1:
            return self._action_lookup.get("noop", 0)

        if not hasattr(self, "_recent_positions"):
            self._recent_positions: List[Tuple[int, int]] = []
            self._max_recent_positions: int = 10

        cur = (state.agent_row, state.agent_col)
        if not self._recent_positions or self._recent_positions[-1] != cur:
            self._recent_positions.append(cur)
            if len(self._recent_positions) > self._max_recent_positions:
                self._recent_positions.pop(0)

        # Ensure move indices initialized
        if not hasattr(self, "_MOVE_N"):
            self._action_to_dir(-999)

        # Horizontal sweep based on row parity
        row_parity = state.agent_row % 2
        preferred_dir = self._MOVE_E if row_parity == 0 else self._MOVE_W

        dr, dc = self._action_to_dir(preferred_dir)
        nr, nc = state.agent_row + (dr or 0), state.agent_col + (dc or 0)
        if preferred_dir in self._MOVE_SET and self._is_valid_position(nr, nc) and self._is_cell_passable(nr, nc):
            return preferred_dir

        # Try moving down a row
        down_r, down_c = state.agent_row + 1, state.agent_col
        if self._is_valid_position(down_r, down_c) and self._is_cell_passable(down_r, down_c):
            return self._MOVE_S if self._MOVE_S != -1 else self._action_lookup.get("noop", 0)

        # Otherwise pick any reasonable alternative, preferring not-recent cells
        alt = self._find_best_exploration_direction(state)
        return (
            alt if alt is not None else (preferred_dir if preferred_dir != -1 else self._action_lookup.get("noop", 0))
        )

    def _find_best_exploration_direction(self, state: AgentState) -> Optional[int]:
        """Pick a direction toward an in-bounds, passable cell; prefer not-recent."""
        # Ensure move indices initialized
        if not hasattr(self, "_MOVE_N"):
            self._action_to_dir(-999)

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
            if not self._is_cell_passable(nr, nc):
                continue
            score = (
                10
                if not hasattr(self, "_recent_positions") or pos not in self._recent_positions
                else self._recent_positions.index(pos)
            )
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
        if hasattr(x, "__len__"):
            if len(x) > 1:
                return int(x[0])
            elif len(x) == 1:
                return ScriptedAgentPolicyImpl._to_int(x[0])
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

    def __init__(self, env: MettaGridEnv | None = None, device=None, hyperparams: Hyperparameters | None = None):
        self._env = env
        self._hyperparams = hyperparams
        if env is not None:
            self._impl = ScriptedAgentPolicyImpl(env, hyperparams=hyperparams)
        else:
            self._impl = None  # Will be set during reset

    def reset(self, obs, info):
        """Reset policy state."""
        if self._env is None and "env" in info:
            self._env = info["env"]
        if self._impl is None:
            if self._env is None:
                raise RuntimeError("ScriptedAgentPolicy needs env - provide during __init__ or in info['env']")
            self._impl = ScriptedAgentPolicyImpl(self._env, hyperparams=self._hyperparams)

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        if self._impl is None:
            raise RuntimeError("Policy not initialized - call reset() first")
        return StatefulAgentPolicy(self._impl, agent_id)
