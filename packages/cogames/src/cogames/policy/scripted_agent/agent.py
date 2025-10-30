"""Enhanced scripted agent with consolidated constants and cleaner object layout."""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from cogames.policy.interfaces import AgentPolicy, Policy, StatefulAgentPolicy, StatefulPolicyImpl
from cogames.policy.scripted_agent.hyperparameters import Hyperparameters
from cogames.policy.scripted_agent.navigator import Navigator
from cogames.policy.scripted_agent.phase_controller import Context, GamePhase, create_controller
from mettagrid import MettaGridAction, MettaGridEnv, MettaGridObservation, dtype_actions

logger = logging.getLogger("cogames.policy.scripted_agent")

# =============================================================================
# GLOBAL CONSTANTS (single source of truth)
# =============================================================================


class C:
    # Inventory requirements
    REQ_CARBON: int = 20
    REQ_OXYGEN: int = 20
    REQ_SILICON: int = 50
    REQ_ENERGY: int = 20

    # Heart detection
    HEART_FEATURE: str = "inv:heart"
    HEART_SENTINEL_FIRST_FIELD: int = 85  # 0x55

    # Observation window (must match env)
    OBS_H: int = 11
    OBS_W: int = 11
    OBS_HR: int = OBS_H // 2
    OBS_WR: int = OBS_W // 2

    # Occupancy encoding
    OCC_UNKNOWN, OCC_FREE, OCC_OBSTACLE = 0, 1, 2

    # Scoring weights
    W_DISTANCE: float = 0.7
    W_EFFICIENCY: float = 0.3

    # Thresholds / heuristics
    DEFAULT_DEPLETION_THRESHOLD: float = 0.25
    LOW_DEPLETION_THRESHOLD: float = 0.25
    CLIPPED_SCORE_PENALTY: float = 0.5
    WAIT_IF_COOLDOWN_LEQ: int = 3  # try-use when <= this
    ROTATE_COOLDOWN_LT: int = 3  # consider alternatives if remaining < this
    ALT_ROTATE_RADIUS: int = 7
    PATIENCE_STEPS: int = 12  # how long to idle when waiting on cooldown
    RECHARGE_BUFFER: float = 5.0
    GATHER_BUFFER_SMALL: float = 10.0
    GATHER_BUFFER_LARGE: float = 5.0
    TASK_ENERGY_SILICON: int = 50

    # Planner knobs
    USE_ASTAR: bool = True
    ASTAR_THRESHOLD: int = 20
    FRONTIER_RADIUS_AROUND_HOME: int = 50
    FRONTIER_SPAWN_RADIUS: int = 30

    # Default (learned) cooldown fallbacks by resource
    DEFAULT_COOLDOWNS: Dict[str, int] = {
        "germanium": 0,
        "silicon": 0,
        "carbon": 10,
        "oxygen": 100,
        "charger": 10,
    }


class FeatureNames:
    TYPE_ID = "type_id"
    CONVERTING = "converting"
    COOLDOWN_REMAINING = "cooldown_remaining"
    CLIPPED = "clipped"
    REMAINING_USES = "remaining_uses"
    INV_CARBON = "inv:carbon"
    INV_OXYGEN = "inv:oxygen"
    INV_GERMANIUM = "inv:germanium"
    INV_SILICON = "inv:silicon"
    INV_ENERGY = "inv:energy"
    GLOBAL_LAST_REWARD = "global:last_reward"


class StationMaps:
    # Which glyph to use per station; phase-specific overrides handled separately
    STATION_TO_GLYPH: Dict[str, str] = {
        "charger": "charger",
        "carbon_extractor": "carbon",
        "oxygen_extractor": "oxygen",
        "germanium_extractor": "germanium",
        "silicon_extractor": "silicon",
        "assembler": "heart",
        "chest": "chest",
    }

    # Which game object stations produce which resource
    STATION_TO_RESOURCE: Dict[str, str] = {
        "carbon_extractor": "carbon",
        "oxygen_extractor": "oxygen",
        "germanium_extractor": "germanium",
        "silicon_extractor": "silicon",
        "charger": "charger",
    }

    # Phase → desired station (high-level)
    PHASE_TO_STATION: Dict[GamePhase, Optional[str]] = {
        GamePhase.GATHER_GERMANIUM: "germanium_extractor",
        GamePhase.GATHER_SILICON: "silicon_extractor",
        GamePhase.GATHER_CARBON: "carbon_extractor",
        GamePhase.GATHER_OXYGEN: "oxygen_extractor",
        GamePhase.ASSEMBLE_HEART: "assembler",
        GamePhase.DEPOSIT_HEART: "chest",
        GamePhase.RECHARGE: "charger",
        GamePhase.UNCLIP_STATION: None,
        GamePhase.CRAFT_DECODER: "assembler",
    }

    # Phase-specific glyph overrides (e.g., crafting)
    PHASE_TO_GLYPH: Dict[GamePhase, str] = {
        GamePhase.CRAFT_DECODER: "gear",
    }


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class ExtractorInfo:
    position: Tuple[int, int]
    resource_type: str  # "carbon", "oxygen", "germanium", "silicon", "charger"
    station_name: str
    last_used_step: int = -1
    total_harvests: int = 0
    total_output: int = 0
    uses_remaining_fraction: float = 1.0
    observed_cooldown_remaining: int = 0
    observed_converting: bool = False
    is_clipped: bool = False
    permanently_depleted: bool = False
    learned_cooldown: Optional[int] = None
    hyperparams: Optional[Hyperparameters] = None

    def is_available(self, step: int, cooldown_fn) -> bool:
        return not (self.permanently_depleted or self.is_clipped) and cooldown_fn(self, step) <= 0

    def is_depleted(self) -> bool:
        return self.permanently_depleted or self.uses_remaining_fraction <= 0.05

    def is_low(self, thr: float | None = None) -> bool:
        if thr is None and self.hyperparams and hasattr(self.hyperparams, "depletion_threshold"):
            thr = self.hyperparams.depletion_threshold
        if thr is None:
            thr = C.DEFAULT_DEPLETION_THRESHOLD
        return self.uses_remaining_fraction <= thr

    def avg_output(self) -> float:
        return 0.0 if self.total_harvests == 0 else self.total_output / self.total_harvests

    def update_after_use(self, output: int, step: int):
        self.last_used_step = step
        self.total_harvests += 1
        self.total_output += output
        # Heuristic: 50 baseline uses
        self.uses_remaining_fraction = max(0.0, 1.0 - self.total_harvests / 50.0)


class ExtractorMemory:
    def __init__(self, hyperparams: Hyperparameters | None = None):
        self._extractors: Dict[str, List[ExtractorInfo]] = defaultdict(list)
        self._by_position: Dict[Tuple[int, int], ExtractorInfo] = {}
        self.hyperparams = hyperparams

    def add_extractor(self, pos: Tuple[int, int], resource_type: str, station_name: str) -> ExtractorInfo:
        if pos in self._by_position:
            logger.debug(f"[Mem] Extractor at {pos} already known")
            return self._by_position[pos]
        e = ExtractorInfo(pos, resource_type, station_name, hyperparams=self.hyperparams)
        self._extractors[resource_type].append(e)
        self._by_position[pos] = e
        logger.info(f"Discovered {resource_type} extractor at {pos}")
        logger.info(f"[Mem] {len(self._extractors[resource_type])} total for {resource_type}")
        return e

    def get_by_type(self, t: str) -> List[ExtractorInfo]:
        return self._extractors[t]

    def get_all(self) -> List[ExtractorInfo]:
        return [e for lst in self._extractors.values() for e in lst]

    def get_at_position(self, pos: Tuple[int, int]) -> Optional[ExtractorInfo]:
        return self._by_position.get(pos)

    def find_best_extractor(
        self, resource: str, cur: Tuple[int, int], step: int, cooldown_fn
    ) -> Optional[ExtractorInfo]:
        def md(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        cands = [e for e in self.get_by_type(resource) if not e.is_depleted() and e.is_available(step, cooldown_fn)]
        if not cands:
            logger.info(f"[Find] No available {resource} extractors (known={len(self.get_by_type(resource))})")
            return None

        def score(e: ExtractorInfo) -> float:
            distance_score = 1.0 / (1.0 + md(e.position, cur))
            eff = e.avg_output()
            efficiency_score = (eff / 50.0) if eff > 0 else 0.5
            depletion = C.CLIPPED_SCORE_PENALTY if e.is_clipped else 1.0
            base = C.W_DISTANCE * distance_score + C.W_EFFICIENCY * efficiency_score
            if e.is_low(C.LOW_DEPLETION_THRESHOLD):
                base *= 0.5
            return base * depletion

        return max(cands, key=score)


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
    # Tracking
    hearts_assembled: int = 0
    wait_counter: int = 0
    just_deposited: bool = False
    # Position
    agent_row: int = -1
    agent_col: int = -1
    # Home + critical stations
    home_base_row: int = -1
    home_base_col: int = -1
    assembler_discovered: bool = False
    chest_discovered: bool = False
    # Waiting/targets
    waiting_since_step: int = -1
    wait_target: Optional[Tuple[int, int]] = None
    unclip_target: Optional[Tuple[int, int]] = None
    unclip_recipes: Dict[str, str] = field(default_factory=dict)
    # Reactive clipping detection
    blocked_by_clipped_extractor: Optional[Tuple[int, int]] = (
        None  # Position of clipped extractor blocking current phase
    )
    # Exploration goal tracking
    explore_goal: Optional[str] = None  # Why we're exploring: "find_charger", "find_assembler", "find_extractor", "find_unclipped", "unstuck"
    # Progress bookkeeping
    phase_entry_step: int = 0
    phase_entry_inventory: Dict[str, int] = field(default_factory=dict)
    unobtainable_resources: Set[str] = field(default_factory=set)
    resource_gathering_start: Dict[str, int] = field(default_factory=dict)
    resource_progress_tracking: Dict[str, int] = field(default_factory=dict)
    phase_visit_count: Dict[str, int] = field(default_factory=dict)
    # Misc
    step_count: int = 0
    last_heart: int = 0
    stuck_counter: int = 0
    last_reward: float = 0.0
    total_reward: float = 0.0


# =============================================================================
# IMPLEMENTATION
# =============================================================================


class ScriptedAgentPolicyImpl(StatefulPolicyImpl[AgentState]):
    def __init__(self, env: MettaGridEnv, hyperparams: Hyperparameters | None = None):
        self._env = env
        self.hyperparams = hyperparams or Hyperparameters()
        self.phase_controller = create_controller(GamePhase.GATHER_GERMANIUM)

        # Names / lookups
        self._action_names: List[str] = env.action_names
        self._object_type_names: List[str] = env.object_type_names
        self._action_lookup: Dict[str, int] = {n: i for i, n in enumerate(self._action_names)}
        feats = env.observation_features
        self._fid: Dict[str, int] = {f.name: f.id for f in feats.values()}

        # Moves
        self._MOVE_N = self._action_lookup.get("move_north", -1)
        self._MOVE_S = self._action_lookup.get("move_south", -1)
        self._MOVE_E = self._action_lookup.get("move_east", -1)
        self._MOVE_W = self._action_lookup.get("move_west", -1)
        self._MOVE_SET = {self._MOVE_N, self._MOVE_S, self._MOVE_E, self._MOVE_W}

        from cogames.cogs_vs_clips.vibes import VIBES

        self._glyph_name_to_id = {v.name: i for i, v in enumerate(VIBES)}

        # Station mappings
        self._station_to_glyph = StationMaps.STATION_TO_GLYPH
        self._phase_to_glyph = StationMaps.PHASE_TO_GLYPH
        self._phase_to_station = StationMaps.PHASE_TO_STATION
        self._station_to_resource_type = StationMaps.STATION_TO_RESOURCE

        # Object ids
        self._type_id_to_station: Dict[int, str] = {}
        self._wall_type_id: Optional[int] = None
        for type_id, name in enumerate(env.object_type_names):
            if name == "wall":
                self._wall_type_id = type_id
            elif name and not name.startswith("agent"):
                self._type_id_to_station[type_id] = name

        # Map geom
        self._map_h, self._map_w = env.c_env.map_height, env.c_env.map_width

        # Adaptive exploration
        import math

        base = self.hyperparams.exploration_phase_steps
        self._explore_steps = int(base * math.sqrt((self._map_h * self._map_w) / 1600.0))
        logger.info(f"[Exploration] steps={self._explore_steps} for map {self._map_h}x{self._map_w}")

        # World state
        self._station_positions: Dict[str, Tuple[int, int]] = {}
        self._visited_cells: set[Tuple[int, int]] = set()
        self._occ = [[C.OCC_UNKNOWN for _ in range(self._map_w)] for _ in range(self._map_h)]
        self._prev_pos: Optional[Tuple[int, int]] = None
        self._last_action_idx: Optional[int] = None
        self._last_attempt_was_use = False

        # Components
        self.extractor_memory = ExtractorMemory(hyperparams=self.hyperparams)
        self._unclip_recipes = self._load_unclip_recipes_from_config()
        self.navigator = Navigator(self._map_h, self._map_w)
        self._cached_state: Optional[AgentState] = None

        logger.info("Scripted agent ready (visual discovery + frontier exploration)")

    # ------------------- Recharge thresholds (derived) -------------------
    @property
    def RECHARGE_START(self) -> int:
        m = max(self._map_h, self._map_w)
        return self.hyperparams.recharge_start_large if m >= 50 else self.hyperparams.recharge_start_small

    @property
    def RECHARGE_STOP(self) -> int:
        m = max(self._map_h, self._map_w)
        return self.hyperparams.recharge_stop_large if m >= 50 else self.hyperparams.recharge_stop_small

    # ----------------------------- Core loop -----------------------------
    def agent_state(self) -> AgentState:
        return AgentState()

    def _update_extractor_after_use(
        self,
        pos: Tuple[int, int],
        s: AgentState,
        resource_gained: int,
        resource_type: str,
    ) -> None:
        """Update memory stats for the extractor at `pos` after a successful harvest."""
        ex = self.extractor_memory.get_at_position(pos)
        if not ex:
            return

        ex.update_after_use(resource_gained, s.step_count)
        avg = ex.avg_output()
        if avg > 0:
            logger.debug(
                f"[Extractor] {resource_type} at {pos}: +{resource_gained} (avg={avg:.1f}, uses={ex.total_harvests})"
            )

    def _update_inventory(self, obs: MettaGridObservation, s: AgentState) -> None:
        """Read inventory strictly from observation tokens."""
        s.carbon = self._read_int_feature(obs, FeatureNames.INV_CARBON)
        s.oxygen = self._read_int_feature(obs, FeatureNames.INV_OXYGEN)
        s.germanium = self._read_int_feature(obs, FeatureNames.INV_GERMANIUM)
        s.silicon = self._read_int_feature(obs, FeatureNames.INV_SILICON)
        s.energy = self._read_int_feature(obs, FeatureNames.INV_ENERGY)
        # Heart presence uses the sentinel rule (first field == 0x55)
        s.heart = 1 if self._has_heart_from_obs(obs) else 0

        # Unclip items (not in FeatureNames on purpose; keep raw keys)
        s.decoder = self._read_int_feature(obs, "inv:decoder")
        s.modulator = self._read_int_feature(obs, "inv:modulator")
        s.resonator = self._read_int_feature(obs, "inv:resonator")
        s.scrambler = self._read_int_feature(obs, "inv:scrambler")

        # Initialize recipe knowledge once
        if not s.unclip_recipes:
            s.unclip_recipes = self._unclip_recipes

    def step_with_state(
        self, obs: MettaGridObservation, state: Optional[AgentState]
    ) -> tuple[MettaGridAction, Optional[AgentState]]:
        s = (
            self._cached_state
            if (state is None or (state.step_count == 0 and self._cached_state is not None))
            else state
        )
        if s is None:
            s = self.agent_state()
        s.step_count += 1
        self._cached_state = s

        self._update_inventory(obs, s)
        self._update_agent_position(s)
        self._update_rewards(obs, s)

        # Home
        if s.home_base_row == -1 and s.agent_row >= 0:
            s.home_base_row, s.home_base_col = s.agent_row, s.agent_col
            logger.info(f"[Init] Home base: ({s.home_base_row},{s.home_base_col})")

        self._mark_cell(s.agent_row, s.agent_col, C.OCC_FREE)
        self._discover_stations_from_observation(obs, s)
        self._update_wall_knowledge(s)

        # Track extractor usage if inventory changed
        inv_changes = self._detect_inventory_changes(s)
        if inv_changes and s.agent_row >= 0:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                pos = (s.agent_row + dr, s.agent_col + dc)
                e = self.extractor_memory.get_at_position(pos)
                if e and inv_changes.get(e.resource_type, 0) > 0:
                    self._update_extractor_after_use(pos, s, inv_changes[e.resource_type], e.resource_type)
                    break

        if s.agent_row >= 0 and s.agent_col >= 0:
            self._visited_cells.add((s.agent_row, s.agent_col))

        # Deposit detection
        if (s.last_heart > 0 and s.heart == 0) or (s.current_phase == GamePhase.DEPOSIT_HEART and s.last_reward > 0):
            s.hearts_assembled += 1
            s.wait_counter = 0
            s.current_phase = GamePhase.GATHER_GERMANIUM
            s.just_deposited = True
            logger.info(f"[Deposit] Heart deposited -> total={s.hearts_assembled}")

        # Phase selection via controller
        old_phase = s.current_phase
        s.current_phase = self._determine_phase(s, obs)
        if s.current_phase != old_phase:
            s.phase_entry_step = s.step_count
            s.phase_entry_inventory = {
                "germanium": s.germanium,
                "silicon": s.silicon,
                "carbon": s.carbon,
                "oxygen": s.oxygen,
                "decoder": s.decoder,
            }
            name = s.current_phase.name
            if name.startswith("GATHER_"):
                r = name.replace("GATHER_", "").lower()
                s.phase_visit_count[r] = s.phase_visit_count.get(r, 0) + 1
                s.resource_progress_tracking.setdefault(r, getattr(s, r))
                s.resource_gathering_start[r] = s.step_count

        action_idx = self._execute_phase(s)
        self._last_action_idx = action_idx
        self._prev_pos = (s.agent_row, s.agent_col)
        s.last_heart = s.heart
        return dtype_actions.type(action_idx), s

    # ------------------------ Phase determination ------------------------
    def _determine_phase(self, s: AgentState, obs: MettaGridObservation) -> GamePhase:
        ctx = Context(obs=obs, env=self._env, step=s.step_count)
        # Make available to guards either via ctx or env
        ctx.policy_impl = self
        if not hasattr(self._env, "policy_impl"):
            self._env.policy_impl = self
        return self.phase_controller.maybe_transition(s, ctx, logger)

    # ---------------------- Phase → concrete action ----------------------
    def _execute_phase(self, s: AgentState) -> int:
        if s.current_phase == GamePhase.EXPLORE:
            self._last_attempt_was_use = False
            plan = self._plan_to_frontier_action(s)
            return plan if plan is not None else self._explore_simple(s)

        if s.current_phase == GamePhase.UNCLIP_STATION:
            return self._do_unclip(s)

        if s.current_phase == GamePhase.CRAFT_DECODER:
            return self._do_craft_decoder(s)

        # Gathering / targeting
        station = self._phase_to_station.get(s.current_phase)
        if not station:
            return self._action_lookup.get("noop", 0)

        # Glyph selection
        need_glyph = self._phase_to_glyph.get(s.current_phase, self._station_to_glyph.get(station, "default"))
        if s.current_glyph != need_glyph:
            s.current_glyph = need_glyph
            s.wait_counter = 0
            gid = self._glyph_name_to_id.get(need_glyph, 0)
            return self._action_lookup.get(f"change_glyph_{gid}", self._action_lookup.get("noop", 0))

        # Choose target for gathering phases via extractor memory
        target = None
        gathering = s.current_phase in [
            GamePhase.GATHER_CARBON,
            GamePhase.GATHER_OXYGEN,
            GamePhase.GATHER_GERMANIUM,
            GamePhase.GATHER_SILICON,
            GamePhase.RECHARGE,
        ]
        if gathering and s.agent_row != -1:
            target = self._find_best_extractor_for_phase(s.current_phase, s)
            if target is None:
                # No extractor available - phase controller will transition to EXPLORE
                logger.info(f"[Phase] {s.current_phase.value}: no available extractors, waiting for phase transition")
                return self._action_lookup.get("noop", 0)
            logger.info(f"[Phase] {s.current_phase.value}: using extractor at {target}")

        # Fallback to known station position
        if target is None:
            target = self._station_positions.get(station)
            if not target or s.agent_row == -1:
                self._last_attempt_was_use = False
                plan = self._plan_to_frontier_action(s)
                return plan if plan is not None else self._explore_simple(s)

        # Navigate
        res = self.navigator.navigate_to(
            start=(s.agent_row, s.agent_col),
            target=target,
            occupancy_map=self._occ,
            optimistic=True,
            use_astar=C.USE_ASTAR,
            astar_threshold=C.ASTAR_THRESHOLD,
        )

        if res.is_adjacent:
            e = self.extractor_memory.get_at_position(target)
            if e and s.wait_target == target:
                rem = self.cooldown_remaining(e, s.step_count)
                if rem > self.hyperparams.wait_if_cooldown_leq:  # hyperparam respected
                    if self._exists_viable_alternative(e.resource_type, s, C.ALT_ROTATE_RADIUS):
                        self._last_attempt_was_use = False
                        logger.debug(f"[Wait→Rotate] cooldown~{rem}")
                        return self._navigate_to_best_alternative(e.resource_type, s)
                    if s.waiting_since_step < 0:
                        s.waiting_since_step = s.step_count
                    if s.step_count - s.waiting_since_step <= C.PATIENCE_STEPS:
                        self._last_attempt_was_use = False
                        return self._action_lookup.get("noop", 0)
                    return self._navigate_to_best_alternative(e.resource_type, s)
                s.wait_target = None

            tr, tc = target
            self._last_attempt_was_use = True
            a = self._step_toward(tr - s.agent_row, tc - s.agent_col)
            return a

        if res.next_step:
            nr, nc = res.next_step
            self._last_attempt_was_use = False
            return self._step_toward(nr - s.agent_row, nc - s.agent_col)

        # Stuck → explore
        self._last_attempt_was_use = False
        plan = self._plan_to_frontier_action(s)
        return plan if plan is not None else self._explore_simple(s)

    # ------------------------ Phase helpers (actions) --------------------
    def _do_unclip(self, s: AgentState) -> int:
        tgt = s.unclip_target
        if tgt:
            cur = self.extractor_memory.get_at_position(tgt)
            if not (cur and cur.is_clipped):
                logger.info(f"[Unclip] target {tgt} cleared")
                s.unclip_target = tgt = None

        if not tgt:
            clipped = [e for e in self.extractor_memory.get_all() if e.is_clipped]
            if not clipped:
                self._last_attempt_was_use = False
                plan = self._plan_to_frontier_action(s)
                return plan if plan is not None else self._explore_simple(s)
            cur = (s.agent_row, s.agent_col)
            chosen = min(clipped, key=lambda e: abs(e.position[0] - cur[0]) + abs(e.position[1] - cur[1]))
            tgt = chosen.position
            s.unclip_target = tgt
            logger.info(f"[Unclip] new target {tgt} ({chosen.resource_type})")

        res = self.navigator.navigate_to(
            start=(s.agent_row, s.agent_col),
            target=tgt,
            occupancy_map=self._occ,
            optimistic=True,
            use_astar=C.USE_ASTAR,
            astar_threshold=C.ASTAR_THRESHOLD,
        )
        if res.is_adjacent:
            tr, tc = tgt
            self._last_attempt_was_use = True
            return self._step_toward(tr - s.agent_row, tc - s.agent_col)
        if res.next_step:
            nr, nc = res.next_step
            self._last_attempt_was_use = False
            return self._step_toward(nr - s.agent_row, nc - s.agent_col)

        self._last_attempt_was_use = False
        plan = self._plan_to_frontier_action(s)
        return plan if plan is not None else self._explore_simple(s)

    def _do_craft_decoder(self, s: AgentState) -> int:
        need_glyph = self._phase_to_glyph.get(s.current_phase, "gear")
        if s.current_glyph != need_glyph:
            s.current_glyph = need_glyph
            s.wait_counter = 0
            gid = self._glyph_name_to_id.get(need_glyph, 0)
            return self._action_lookup.get(f"change_glyph_{gid}", self._action_lookup.get("noop", 0))

        asm = self._station_positions.get("assembler")
        if not asm:
            self._last_attempt_was_use = False
            plan = self._plan_to_frontier_action(s)
            return plan if plan is not None else self._explore_simple(s)

        res = self.navigator.navigate_to(
            start=(s.agent_row, s.agent_col),
            target=asm,
            occupancy_map=self._occ,
            optimistic=True,
            use_astar=C.USE_ASTAR,
            astar_threshold=C.ASTAR_THRESHOLD,
        )
        if res.is_adjacent:
            tr, tc = asm
            self._last_attempt_was_use = True
            return self._step_toward(tr - s.agent_row, tc - s.agent_col)
        if res.next_step:
            nr, nc = res.next_step
            self._last_attempt_was_use = False
            return self._step_toward(nr - s.agent_row, nc - s.agent_col)

        self._last_attempt_was_use = False
        plan = self._plan_to_frontier_action(s)
        return plan if plan is not None else self._explore_simple(s)

    # ----------------------------- Targeting -----------------------------
    def _find_best_extractor_for_phase(self, phase: GamePhase, s: AgentState) -> Optional[Tuple[int, int]]:
        phase_to_resource = {
            GamePhase.GATHER_CARBON: "carbon",
            GamePhase.GATHER_OXYGEN: "oxygen",
            GamePhase.GATHER_GERMANIUM: "germanium",
            GamePhase.GATHER_SILICON: "silicon",
            GamePhase.RECHARGE: "charger",
        }
        r = phase_to_resource.get(phase)
        if not r:
            return None

        logger.info(f"[Find] {r}: {len(self.extractor_memory.get_by_type(r))} known")
        cur = (s.agent_row, s.agent_col)
        best = self.extractor_memory.find_best_extractor(r, cur, s.step_count, self.cooldown_remaining)
        if best is None:
            xs = [e for e in self.extractor_memory.get_by_type(r) if not e.is_depleted()]
            if xs:
                # Check if any are clipped - if so, we need to unclip them
                clipped_extractors = [e for e in xs if e.is_clipped]
                if clipped_extractors:
                    # Set flag to trigger unclipping subprocess
                    closest_clipped = min(
                        clipped_extractors,
                        key=lambda e: abs(e.position[0] - s.agent_row) + abs(e.position[1] - s.agent_col),
                    )
                    s.blocked_by_clipped_extractor = closest_clipped.position
                    logger.info(
                        f"[Find] No available {r} extractors → {len(clipped_extractors)} clipped, need to unclip"
                    )
                    return None

                def est(e):
                    return self.cooldown_remaining(e, s.step_count)

                cand = min(xs, key=est)
                cd = est(cand)
                dist = abs(cand.position[0] - s.agent_row) + abs(cand.position[1] - s.agent_col)
                should_wait = (cd < 10) or (cd < 20 and dist < 5) or (cd < 100 and dist <= 1)
                if should_wait:
                    s.wait_target = cand.position
                    if s.waiting_since_step < 0:
                        s.waiting_since_step = s.step_count
                    return cand.position
            logger.info(f"[Find] No available {r} extractors (known={len(self.extractor_memory.get_by_type(r))})")
            return None

        # Energy feasibility
        task_energy = C.TASK_ENERGY_SILICON if r == "silicon" else 0
        if not self._can_reach_safely(best.position, s, task_energy, is_recharge=(r == "charger")):
            logger.warning(f"[Find] Insufficient energy to reach {r} at {best.position}")
            return None
        return best.position

    # --------------------------- Exploration -----------------------------
    def _plan_to_frontier_action(self, s: AgentState) -> Optional[int]:
        t = self._choose_frontier(s)
        if not t:
            return None
        tr, tc = t
        sr, sc = s.agent_row, s.agent_col
        if abs(tr - sr) + abs(tc - sc) == 1:
            return self._step_toward(tr - sr, tc - sc)

        neighbors = [
            (nr, nc) for nr, nc in self._neighbors4(tr, tc) if self._occ[nr][nc] == C.OCC_FREE and (nr, nc) != (sr, sc)
        ]
        if not neighbors:
            return None
        neighbors.sort(key=lambda p: abs(p[0] - sr) + abs(p[1] - sc))
        for goal in neighbors:
            step = self._bfs_next_step_occ((sr, sc), goal)
            if step is not None:
                return self._step_toward(step[0] - sr, step[1] - sc)
        return None

    def _choose_frontier(self, s: AgentState) -> Optional[Tuple[int, int]]:
        if s.agent_row < 0:
            return None
        start = (s.agent_row, s.agent_col)
        fronts = set(self._compute_frontiers())
        if not fronts:
            return None

        # radius around home
        center = (s.home_base_row, s.home_base_col) if s.home_base_row >= 0 else start
        if C.FRONTIER_RADIUS_AROUND_HOME > 0:
            f2 = {
                (r, c) for (r, c) in fronts if abs(r - center[0]) + abs(c - center[1]) <= C.FRONTIER_RADIUS_AROUND_HOME
            }
            if f2:
                fronts = f2

        # Find assembler/chest sooner: spawn and center bias
        if not s.assembler_discovered or not s.chest_discovered:
            if s.home_base_row >= 0:
                cand = [
                    (fr, fc)
                    for (fr, fc) in fronts
                    if abs(fr - s.home_base_row) + abs(fc - s.home_base_col) <= C.FRONTIER_SPAWN_RADIUS
                ]
                if cand:
                    return min(cand, key=lambda p: abs(p[0] - start[0]) + abs(p[1] - start[1]))
            map_center = (self._map_h // 2, self._map_w // 2)
            q, seen, reach = deque([(start, 0)]), {start}, []
            while q:
                (r, c), d = q.popleft()
                for nr, nc in self._neighbors4(r, c):
                    if (nr, nc) in fronts:
                        reach.append(((nr, nc), d + 1))
                for nr, nc in self._neighbors4(r, c):
                    if (nr, nc) in seen or self._occ[nr][nc] != C.OCC_FREE:
                        continue
                    seen.add((nr, nc))
                    q.append(((nr, nc), d + 1))
            best, best_s, denom = None, float("inf"), max(self._map_h, self._map_w)
            for (fr, fc), bfsd in reach:
                s2 = 0.5 * (bfsd / denom) + 0.5 * ((abs(fr - map_center[0]) + abs(fc - map_center[1])) / denom)
                if s2 < best_s:
                    best_s, best = s2, (fr, fc)
            if best:
                return best

        # Nearest by BFS
        return self._choose_frontier_bfs(start, fronts)

    def _choose_frontier_bfs(self, start: Tuple[int, int], fronts: Set[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        q, seen = deque([start]), {start}
        while q:
            r, c = q.popleft()
            for nr, nc in self._neighbors4(r, c):
                if (nr, nc) in fronts:
                    return (nr, nc)
            for nr, nc in self._neighbors4(r, c):
                if (nr, nc) in seen or self._occ[nr][nc] != C.OCC_FREE:
                    continue
                seen.add((nr, nc))
                q.append((nr, nc))
        return None

    def _compute_frontiers(self) -> List[Tuple[int, int]]:
        res = []
        for r in range(self._map_h):
            for c in range(self._map_w):
                if self._occ[r][c] != C.OCC_UNKNOWN:
                    continue
                for nr, nc in self._neighbors4(r, c):
                    if self._occ[nr][nc] == C.OCC_FREE:
                        res.append((r, c))
                        break
        return res

    # ------------------------------ Movement -----------------------------
    def _bfs_next_step_occ(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        return self._bfs_next_step(start, goal, optimistic=False)

    def _bfs_next_step_optimistic(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        return self._bfs_next_step(start, goal, optimistic=True)

    def _bfs_next_step(
        self, start: Tuple[int, int], goal: Tuple[int, int], optimistic: bool
    ) -> Optional[Tuple[int, int]]:
        if start == goal:
            return start
        q, parent = deque([start]), {start: None}
        while q:
            r, c = q.popleft()
            for nr, nc in self._neighbors4(r, c):
                if (nr, nc) in parent or not self._is_cell_passable(nr, nc, optimistic):
                    continue
                parent[(nr, nc)] = (r, c)
                if (nr, nc) == goal:
                    return self._reconstruct_first_step(parent, start, goal)
                q.append((nr, nc))
        return None

    def _reconstruct_first_step(
        self, parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]], start: Tuple[int, int], goal: Tuple[int, int]
    ) -> Tuple[int, int]:
        step = goal
        while parent.get(step) is not None and parent[step] != start:
            step = parent[step]
        return step

    def _is_cell_passable(self, r: int, c: int, optimistic: bool = False) -> bool:
        cell = self._occ[r][c]
        return (cell != C.OCC_OBSTACLE) if optimistic else (cell == C.OCC_FREE)

    def _action_to_dir(self, idx: int) -> Tuple[Optional[int], Optional[int]]:
        if idx == self._MOVE_N:
            return -1, 0
        if idx == self._MOVE_S:
            return 1, 0
        if idx == self._MOVE_E:
            return 0, 1
        if idx == self._MOVE_W:
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

    # ------------------------- Energy feasibility ------------------------
    def _can_reach_safely(
        self,
        target: Tuple[int, int],
        s: AgentState,
        task_energy: int = 0,
        is_recharge: bool = False,
        regen_per_step: Optional[float] = None,
    ) -> bool:
        if s.agent_row == -1:
            return True
        rps = 1.0 if regen_per_step is None else regen_per_step
        d = abs(target[0] - s.agent_row) + abs(target[1] - s.agent_col)
        net = 1.0 - rps
        if is_recharge:
            return s.energy >= max(0.0, d * net) + C.RECHARGE_BUFFER or net <= 0
        trip_steps = d * (2 if task_energy < C.TASK_ENERGY_SILICON else 1)
        req = (
            max(0.0, trip_steps * net)
            + task_energy
            + (C.GATHER_BUFFER_SMALL if task_energy < C.TASK_ENERGY_SILICON else C.GATHER_BUFFER_LARGE)
        )
        return True if net <= 0 else s.energy >= req

    # ------------------------------ Waiting/Alt --------------------------
    def _exists_viable_alternative(self, resource: str, s: AgentState, radius: int) -> bool:
        cur = (s.agent_row, s.agent_col)
        for e in self.extractor_memory.get_by_type(resource):
            if e.position == s.wait_target or e.is_low(C.LOW_DEPLETION_THRESHOLD):
                continue
            if (
                abs(e.position[0] - cur[0]) + abs(e.position[1] - cur[1]) <= radius
                and self.cooldown_remaining(e, s.step_count) < C.ROTATE_COOLDOWN_LT
            ):
                return True
        return False

    def _navigate_to_best_alternative(self, resource: str, s: AgentState) -> int:
        cur = (s.agent_row, s.agent_col)
        alts = [
            e
            for e in self.extractor_memory.get_by_type(resource)
            if e.position != s.wait_target
            and not e.is_low(C.LOW_DEPLETION_THRESHOLD)
            and self.cooldown_remaining(e, s.step_count) < C.ROTATE_COOLDOWN_LT
        ]
        if not alts:
            return self._explore_simple(s)
        best = min(alts, key=lambda e: abs(e.position[0] - cur[0]) + abs(e.position[1] - cur[1]))
        res = self.navigator.navigate_to(
            start=cur,
            target=best.position,
            occupancy_map=self._occ,
            optimistic=True,
            use_astar=C.USE_ASTAR,
            astar_threshold=C.ASTAR_THRESHOLD,
        )
        if res.next_step:
            nr, nc = res.next_step
            self._last_attempt_was_use = False
            return self._step_toward(nr - s.agent_row, nc - s.agent_col)
        return self._explore_simple(s)

    # -------------------------- Observation layer ------------------------
    def _discover_stations_from_observation(self, obs: MettaGridObservation, s: AgentState) -> None:
        if s.agent_row == -1:
            return
        f = self._fid
        f_type = f.get(FeatureNames.TYPE_ID, 0)
        f_conv = f.get(FeatureNames.CONVERTING, 5)
        f_cd = f.get(FeatureNames.COOLDOWN_REMAINING, 14)
        f_clip = f.get(FeatureNames.CLIPPED, 15)
        f_uses = f.get(FeatureNames.REMAINING_USES, 16)

        pos_feats: Dict[Tuple[int, int], Dict[str, int]] = {}
        t2i = self._to_int

        for tok in obs:
            feature_id = t2i(tok[1])
            packed = t2i(tok[0])
            obs_r, obs_c = packed >> 4, packed & 0x0F
            r, c = obs_r - C.OBS_HR + s.agent_row, obs_c - C.OBS_WR + s.agent_col
            if not self._is_valid_position(r, c):
                continue
            pf = pos_feats.setdefault((r, c), {})
            if feature_id == f_type:
                pf["type_id"] = t2i(tok[2])
            elif feature_id == f_conv:
                pf["converting"] = t2i(tok[2])
            elif feature_id == f_cd:
                pf["cooldown_remaining"] = t2i(tok[2])
            elif feature_id == f_clip:
                pf["clipped"] = t2i(tok[2])
            elif feature_id == f_uses:
                pf["remaining_uses"] = t2i(tok[2])

        for (r, c), feats in pos_feats.items():
            tid = feats.get("type_id")
            if tid is None:
                continue
            if tid == self._wall_type_id:
                self._mark_cell(r, c, C.OCC_OBSTACLE)
                continue

            if tid in self._type_id_to_station:
                station = self._type_id_to_station[tid]
                self._mark_cell(r, c, C.OCC_OBSTACLE)

                if station not in self._station_positions:
                    self._station_positions[station] = (r, c)
                    if station == "assembler":
                        s.assembler_discovered = True
                    elif station == "chest":
                        s.chest_discovered = True

                rtype = self._station_to_resource_type.get(station)
                if rtype:
                    ex = self.extractor_memory.add_extractor((r, c), rtype, station)
                    if ex.learned_cooldown is None:
                        ex.learned_cooldown = C.DEFAULT_COOLDOWNS.get(rtype, 10)
                    if "converting" in feats:
                        ex.observed_converting = bool(feats["converting"])
                    if "clipped" in feats:
                        was = ex.is_clipped
                        ex.is_clipped = bool(feats["clipped"])
                        if not ex.is_clipped and was and s.unclip_target == (r, c):
                            s.unclip_target = None
                    else:
                        if ex.is_clipped:
                            ex.is_clipped = False
                            if s.unclip_target == (r, c):
                                s.unclip_target = None
                    if "cooldown_remaining" in feats:
                        cv = feats["cooldown_remaining"]
                        ex.observed_cooldown_remaining = cv
                        if ex.last_used_step >= 0:
                            elapsed = s.step_count - ex.last_used_step
                            if cv > 0 and (ex.learned_cooldown is None or elapsed < 5):
                                ex.learned_cooldown = elapsed + cv
                            elif cv == 0 and elapsed > 0 and (ex.learned_cooldown is None or elapsed < 20):
                                ex.learned_cooldown = elapsed
                    if "remaining_uses" in feats:
                        v = feats["remaining_uses"]
                        ex.uses_remaining_fraction = min(1.0, v / 50.0)
                        if v == 0 and not ex.permanently_depleted:
                            ex.permanently_depleted = True
                continue

            if not self._object_type_names[tid].startswith("agent"):
                self._mark_cell(r, c, C.OCC_FREE)

        self._mark_cell(s.agent_row, s.agent_col, C.OCC_FREE)

    # ----------------------------- Bookkeeping ---------------------------
    def _update_wall_knowledge(self, s: AgentState) -> None:
        if not self._prev_pos or self._last_action_idx not in self._MOVE_SET:
            return
        cur = (s.agent_row, s.agent_col)
        if cur != self._prev_pos:
            self._mark_cell(*cur, C.OCC_FREE)
            return
        if self._last_attempt_was_use:
            self._last_attempt_was_use = False
            return
        dr, dc = self._action_to_dir(self._last_action_idx)
        if dr is None:
            return
        wr, wc = self._prev_pos[0] + dr, self._prev_pos[1] + dc
        if not self._is_valid_position(wr, wc):
            return
        self._occ[wr][wc] = C.OCC_OBSTACLE

    def _update_agent_position(self, s: AgentState) -> None:
        try:
            for _id, obj in self._env.c_env.grid_objects().items():
                if obj.get("agent_id") == 0:
                    s.agent_row, s.agent_col = obj.get("r", -1), obj.get("c", -1)
                    break
        except Exception:
            pass

    def _update_rewards(self, obs: MettaGridObservation, s: AgentState) -> None:
        try:
            fid = self._fid.get(FeatureNames.GLOBAL_LAST_REWARD)
            if fid is None:
                return
            ti = self._to_int
            for tok in obs:
                if ti(tok[1]) == fid:
                    s.last_reward = float(ti(tok[2]))
                    s.total_reward += s.last_reward
                    break
        except Exception:
            pass

    def _has_heart_from_obs(self, obs: MettaGridObservation) -> bool:
        fid = self._fid.get(C.HEART_FEATURE)
        if fid is None:
            return False
        ti = self._to_int
        for tok in obs:
            if ti(tok[1]) == fid:
                return ti(tok[0]) == C.HEART_SENTINEL_FIRST_FIELD
        return False

    def _read_int_feature(self, obs: MettaGridObservation, name: str) -> int:
        fid = self._fid.get(name)
        if fid is None:
            return 0
        ti = self._to_int
        for tok in obs:
            if ti(tok[1]) == fid:
                return ti(tok[2])
        return 0

    def _detect_inventory_changes(self, s: AgentState) -> Dict[str, int]:
        cur = {
            "carbon": s.carbon,
            "oxygen": s.oxygen,
            "germanium": s.germanium,
            "silicon": s.silicon,
            "energy": s.energy,
        }
        if not hasattr(self, "_prev_inventory"):
            self._prev_inventory = cur.copy()
            return {}
        changes = {
            k: cur[k] - self._prev_inventory.get(k, cur[k])
            for k in cur
            if cur[k] != self._prev_inventory.get(k, cur[k])
        }
        self._prev_inventory = cur.copy()
        return changes

    def cooldown_remaining(self, e: ExtractorInfo, step: int) -> int:
        if e.last_used_step >= 0:
            total = e.learned_cooldown if e.learned_cooldown is not None else 20
            return max(0, total - max(0, step - e.last_used_step))
        return 0

    def _load_unclip_recipes_from_config(self) -> Dict[str, str]:
        """Load unclip recipes and return mapping: clipped_resource → craft_resource.

        E.g., {"oxygen": "carbon"} means "to unclip oxygen, craft decoder from carbon"
        """
        # First, get item → craft_resource mapping from assembler
        item_to_craft_resource: Dict[str, str] = {}
        try:
            cfg = getattr(self._env, "env_cfg", None)
            if cfg and hasattr(cfg, "game") and hasattr(cfg.game, "objects"):
                assembler = cfg.game.objects.get("assembler")
                if assembler and hasattr(assembler, "recipes"):
                    for glyph_seq, protocol in assembler.recipes:
                        if "gear" in glyph_seq:
                            outs, ins = protocol.output_resources, protocol.input_resources
                            for item in outs:
                                if item in ["decoder", "modulator", "resonator", "scrambler"]:
                                    for res in ins:
                                        if res in ["carbon", "oxygen", "germanium", "silicon"]:
                                            item_to_craft_resource[item] = res
                                            break
        except Exception:
            pass

        if not item_to_craft_resource:
            item_to_craft_resource = {
                "decoder": "carbon",
                "modulator": "oxygen",
                "resonator": "silicon",
                "scrambler": "germanium",
            }

        # Now convert to clipped_resource → craft_resource mapping
        # Standard mapping: decoder unclips oxygen, modulator unclips carbon, etc.
        item_to_clipped_resource = {
            "decoder": "oxygen",
            "modulator": "carbon",
            "resonator": "germanium",
            "scrambler": "silicon",
        }

        recipes: Dict[str, str] = {}
        for item, clipped_res in item_to_clipped_resource.items():
            if item in item_to_craft_resource:
                recipes[clipped_res] = item_to_craft_resource[item]

        return recipes

    # ------------------------------- Utils -------------------------------
    @staticmethod
    def _to_int(x) -> int:
        if isinstance(x, int):
            return x
        if hasattr(x, "__len__"):
            if len(x) > 1:
                return int(x[0])
            if len(x) == 1:
                return ScriptedAgentPolicyImpl._to_int(x[0])
        if hasattr(x, "item"):
            return int(x.item())
        return int(x)

    def _is_valid_position(self, r: int, c: int) -> bool:
        return 0 <= r < self._map_h and 0 <= c < self._map_w

    def _mark_cell(self, r: int, c: int, cell_type: int) -> None:
        if self._is_valid_position(r, c):
            self._occ[r][c] = cell_type

    def _neighbors4(self, r: int, c: int) -> List[Tuple[int, int]]:
        res: List[Tuple[int, int]] = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if self._is_valid_position(nr, nc):
                res.append((nr, nc))
        return res

    def _explore_simple(self, s: AgentState) -> int:
        if s.agent_row == -1:
            return self._action_lookup.get("noop", 0)
        # Boustrophedon sweep with tiny memory
        if not hasattr(self, "_recent_positions"):
            self._recent_positions: List[Tuple[int, int]] = []
            self._max_recent_positions = 10
        cur = (s.agent_row, s.agent_col)
        if not self._recent_positions or self._recent_positions[-1] != cur:
            self._recent_positions.append(cur)
            if len(self._recent_positions) > self._max_recent_positions:
                self._recent_positions.pop(0)

        row_parity = s.agent_row % 2
        preferred = self._MOVE_E if row_parity == 0 else self._MOVE_W
        dr, dc = self._action_to_dir(preferred)
        nr, nc = s.agent_row + (dr or 0), s.agent_col + (dc or 0)
        if preferred in self._MOVE_SET and self._is_valid_position(nr, nc) and self._is_cell_passable(nr, nc):
            return preferred

        down_r, down_c = s.agent_row + 1, s.agent_col
        if self._is_valid_position(down_r, down_c) and self._is_cell_passable(down_r, down_c):
            return self._MOVE_S if self._MOVE_S != -1 else self._action_lookup.get("noop", 0)

        # Choose any in-bounds, passable, not-recent direction
        options = [(self._MOVE_N, (-1, 0)), (self._MOVE_S, (1, 0)), (self._MOVE_E, (0, 1)), (self._MOVE_W, (0, -1))]
        best, best_score = None, -1
        for a, (dr, dc) in options:
            nr, nc = s.agent_row + dr, s.agent_col + dc
            if not self._is_valid_position(nr, nc):
                continue
            if not self._is_cell_passable(nr, nc):
                continue
            pos = (nr, nc)
            score = 10 if pos not in getattr(self, "_recent_positions", []) else self._recent_positions.index(pos)
            if score > best_score:
                best_score, best = score, a
        return best if best is not None else (preferred if preferred != -1 else self._action_lookup.get("noop", 0))


# =============================================================================
# PUBLIC POLICY WRAPPER
# =============================================================================


class ScriptedAgentPolicy(Policy):
    def __init__(self, env: MettaGridEnv | None = None, device=None, hyperparams: Hyperparameters | None = None):
        self._env = env
        self._hyperparams = hyperparams
        self._impl = ScriptedAgentPolicyImpl(env, hyperparams=hyperparams) if env is not None else None

    def reset(self, obs, info):
        if self._env is None and "env" in info:
            self._env = info["env"]
        if self._impl is None:
            if self._env is None:
                raise RuntimeError("ScriptedAgentPolicy needs env - provide during __init__ or via info['env']")
            self._impl = ScriptedAgentPolicyImpl(self._env, hyperparams=self._hyperparams)

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        if self._impl is None:
            raise RuntimeError("Policy not initialized - call reset() first")
        return StatefulAgentPolicy(self._impl, agent_id)
