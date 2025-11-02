"""Enhanced scripted agent with consolidated constants and cleaner object layout."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from math import ceil
from typing import Dict, List, Optional, Set, Tuple

from cogames.policy.interfaces import AgentPolicy, Policy, StatefulAgentPolicy, StatefulPolicyImpl
from cogames.policy.scripted_agent.constants import C, FeatureNames, StationMaps
from cogames.policy.scripted_agent.hyperparameters import Hyperparameters
from cogames.policy.scripted_agent.navigator import Navigator
from cogames.policy.scripted_agent.phase_controller import (
    Context,
    GamePhase,
    create_controller,
    has_all_materials,
    have_assembler_discovered,
)
from cogames.policy.scripted_agent.state import AgentState
from mettagrid import MettaGridAction, MettaGridEnv, MettaGridObservation, dtype_actions

logger = logging.getLogger("cogames.policy.scripted_agent")


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
        self._resource_to_station: Dict[str, str] = {
            resource: station for station, resource in self._station_to_resource_type.items()
        }
        self._phase_to_resource: Dict[GamePhase, Optional[str]] = {
            phase: self._station_to_resource_type.get(station) for phase, station in self._phase_to_station.items()
        }
        self._resource_to_phase: Dict[str, GamePhase] = {
            resource: phase for phase, resource in self._phase_to_resource.items() if resource is not None
        }
        self._gather_phases: Set[GamePhase] = {
            GamePhase.GATHER_GERMANIUM,
            GamePhase.GATHER_SILICON,
            GamePhase.GATHER_CARBON,
            GamePhase.GATHER_OXYGEN,
        }
        self._gather_resources: Tuple[str, ...] = ("germanium", "silicon", "carbon", "oxygen")

        # Shared debug + per-agent bookkeeping caches
        self._prev_inventory: Dict[int, Dict[str, int]] = {}
        self._recent_positions: Dict[int, deque[Tuple[int, int]]] = {}
        self._max_recent_positions: int = 10
        self._status_log_interval: int = 25
        self._resource_focus_counts: Dict[str, int] = defaultdict(int)
        self._num_agents: int = getattr(env, "num_agents", 1)
        self._resource_focus_limits: Dict[str, int] = {
            "oxygen": max(1, ceil(self._num_agents / 4)),
            "silicon": max(2, ceil(self._num_agents / 2)),
            "germanium": max(1, ceil(self._num_agents / 3)),
            "carbon": max(1, ceil(self._num_agents / 3)),
            "assembler": 1,
        }
        self._target_assignments: Dict[int, Tuple[int, int]] = {}
        # Prefer side positions first so chest/charger approaches stay clear
        self._assembler_offsets: list[tuple[int, int]] = [(0, -1), (0, 1), (1, 0), (-1, 0)]
        self._assembler_slot_for_agent: Dict[int, int] = {}
        self._resource_focus_limits["assembler"] = len(self._assembler_offsets)
        self._assembly_signal = {"active": False, "requester": None, "position": None}
        self._assembly_signal_participants: set[int] = set()
        self.recharge_idle_tolerance: int = getattr(
            self.hyperparams, "recharge_idle_tolerance", C.RECHARGE_IDLE_TOLERANCE
        )

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
        self._should_use_probes: bool = max(self._map_h, self._map_w) >= 60
        self._hub_probe_offsets: list[Tuple[int, int]] = self._build_probe_offsets()

        # World state (shared across agents - only station positions)
        self._station_positions: Dict[str, Tuple[int, int]] = {}
        # Per-agent state (occupancy map, navigation state) now in AgentState

        # Components
        self.extractor_memory = ExtractorMemory(hyperparams=self.hyperparams)
        self._unclip_recipes = self._load_unclip_recipes_from_config()
        self._heart_requirements = self._infer_heart_requirements()
        self.navigator = Navigator(self._map_h, self._map_w)
        # Note: _cached_state removed - it was shared across agents and caused bugs in multi-agent

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
    def agent_state(self, agent_id: int = 0) -> AgentState:
        # Initialize per-agent state with fresh occupancy map
        if agent_id == 0:
            self._assembly_signal = {"active": False, "requester": None, "position": None}
            self._assembly_signal_participants.clear()
            self._assembler_slot_for_agent.clear()
        state = AgentState(agent_id=agent_id)
        state.occupancy_map = [[C.OCC_UNKNOWN for _ in range(self._map_w)] for _ in range(self._map_h)]

        # Randomize resource gathering order per agent for multi-agent diversity
        import random

        resources = ["germanium", "silicon", "carbon", "oxygen"]
        # Use agent_id as seed for reproducibility
        rng = random.Random(agent_id + 42)
        state.resource_order = resources.copy()
        rng.shuffle(state.resource_order)

        # Set initial phase based on first resource in order
        phase_map = {
            "germanium": GamePhase.GATHER_GERMANIUM,
            "silicon": GamePhase.GATHER_SILICON,
            "carbon": GamePhase.GATHER_CARBON,
            "oxygen": GamePhase.GATHER_OXYGEN,
        }
        state.current_phase = phase_map.get(state.resource_order[0], GamePhase.GATHER_GERMANIUM)
        state.heart_requirements = dict(self._heart_requirements)
        state.is_leader = agent_id == 0

        return state

    # ----------------------- Resource bookkeeping -----------------------
    def _infer_heart_requirements(self) -> Dict[str, int]:
        defaults = {
            "carbon": C.REQ_CARBON,
            "oxygen": C.REQ_OXYGEN,
            "silicon": C.REQ_SILICON,
            "germanium": 5,
            "energy": C.REQ_ENERGY,
        }
        try:
            cfg = getattr(self._env, "env_cfg", None)
            assembler = getattr(getattr(cfg, "game", None), "objects", {}).get("assembler") if cfg else None
            recipes = getattr(assembler, "recipes", None)
            if not recipes:
                return defaults

            best_inputs: Optional[Dict[str, int]] = None
            best_cost = float("inf")
            for _glyphs, protocol in recipes:
                outputs = getattr(protocol, "output_resources", {}) or {}
                if outputs.get("heart", 0) <= 0:
                    continue
                inputs = {k: int(v) for k, v in (getattr(protocol, "input_resources", {}) or {}).items()}
                total_cost = sum(inputs.values())
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_inputs = inputs

            if not best_inputs:
                return defaults

            inferred = defaults.copy()
            for resource, default in defaults.items():
                inferred[resource] = max(0, best_inputs.get(resource, default))
            return inferred
        except Exception as exc:  # pragma: no cover - defensive path
            logger.debug("[Hearts] Failed to infer heart requirements: %s", exc)
            return defaults

    def _resource_requirements(self, s: AgentState) -> Dict[str, int]:
        base = s.heart_requirements or self._heart_requirements
        return {
            "germanium": max(1, base.get("germanium", 5)),
            "silicon": base.get("silicon", C.REQ_SILICON),
            "carbon": base.get("carbon", C.REQ_CARBON),
            "oxygen": base.get("oxygen", C.REQ_OXYGEN),
        }

    def _resource_deficits(self, s: AgentState) -> Dict[str, int]:
        requirements = self._resource_requirements(s)
        return {res: max(0, required - getattr(s, res, 0)) for res, required in requirements.items()}

    def _next_needed_resource(self, s: AgentState, deficits: Dict[str, int]) -> Optional[str]:
        for resource in s.resource_order:
            if deficits.get(resource, 0) > 0:
                return resource
        for resource in self._gather_resources:
            if deficits.get(resource, 0) > 0:
                return resource
        return None

    def _choose_resource_focus(
        self, s: AgentState, proposed_phase: GamePhase, deficits: Dict[str, int]
    ) -> Optional[str]:
        preferred = self._phase_to_resource.get(proposed_phase)
        if preferred and deficits.get(preferred, 0) > 0 and self._can_focus_resource(preferred):
            return preferred
        next_needed = self._next_needed_resource(s, deficits)
        if next_needed and not self._can_focus_resource(next_needed):
            candidates = [res for res in s.resource_order if deficits.get(res, 0) > 0]
            for resource in candidates:
                if self._can_focus_resource(resource):
                    return resource
            for resource in self._gather_resources:
                if deficits.get(resource, 0) > 0 and self._can_focus_resource(resource):
                    return resource
        if next_needed is not None and self._can_focus_resource(next_needed):
            return next_needed
        return preferred

    def _resolve_gather_target(
        self,
        s: AgentState,
        focus_resource: Optional[str],
        deficits: Dict[str, int],
    ) -> tuple[Optional[str], Optional[Tuple[int, int]]]:
        if s.agent_row == -1:
            return focus_resource, None

        order: List[str] = []
        if focus_resource and deficits.get(focus_resource, 0) > 0:
            order.append(focus_resource)
        for resource in s.resource_order:
            if resource not in order and deficits.get(resource, 0) > 0:
                order.append(resource)
        for resource in self._gather_resources:
            if resource not in order and deficits.get(resource, 0) > 0:
                order.append(resource)

        current = (s.agent_row, s.agent_col)
        assignment_counts = Counter(self._target_assignments.values())
        for resource in order:
            candidates: list[tuple[int, int, float, ExtractorInfo]] = []
            for extractor in self.extractor_memory.get_by_type(resource):
                if extractor.is_depleted() or extractor.is_clipped:
                    continue
                distance = abs(extractor.position[0] - current[0]) + abs(extractor.position[1] - current[1])
                assigned = assignment_counts.get(extractor.position, 0)
                candidates.append((assigned, distance, -extractor.avg_output(), extractor))

            if candidates:
                candidates.sort(key=lambda item: (item[0], item[1], item[2]))
                chosen = candidates[0][3]
                self._target_assignments[s.agent_id] = chosen.position
                s.explore_goal = None
                return resource, chosen.position

            extractor = self.extractor_memory.find_best_extractor(
                resource, current, s.step_count, self.cooldown_remaining
            )
            if extractor is not None:
                self._target_assignments[s.agent_id] = extractor.position
                s.explore_goal = None
                return resource, extractor.position

            if deficits.get(resource, 0) > 0:
                s.explore_goal = f"find_{resource}"
                return resource, None

        if focus_resource and deficits.get(focus_resource, 0) > 0:
            s.explore_goal = f"find_{focus_resource}"
        elif not deficits or all(v <= 0 for v in deficits.values()):
            s.explore_goal = None
        return focus_resource, None

    def _log_agent_status(self, s: AgentState) -> None:
        if self._status_log_interval <= 0 or not logger.isEnabledFor(logging.INFO):
            return
        if s.step_count % self._status_log_interval != 0:
            return
        deficits = self._resource_deficits(s)
        logger.info(
            "[Agent %s] step=%d phase=%s pos=(%d,%d) inv G:%d S:%d C:%d O:%d E:%d heart=%d target=%s deficits=%s",
            s.agent_id,
            s.step_count,
            s.current_phase.name,
            s.agent_row,
            s.agent_col,
            s.germanium,
            s.silicon,
            s.carbon,
            s.oxygen,
            s.energy,
            s.heart,
            s.active_resource_target,
            deficits,
        )

    def _build_probe_offsets(self) -> list[Tuple[int, int]]:
        if not self._should_use_probes:
            return []
        span = max(self._map_h, self._map_w)
        base = max(8, span // 12)
        distances = [base, base * 2]
        offsets: list[Tuple[int, int]] = []
        for dist in distances:
            offsets.extend(
                [
                    (dist, 0),
                    (-dist, 0),
                    (0, dist),
                    (0, -dist),
                    (dist, dist),
                    (dist, -dist),
                    (-dist, dist),
                    (-dist, -dist),
                ]
            )
        # Remove duplicates while preserving order
        unique_offsets: list[Tuple[int, int]] = []
        seen: Set[Tuple[int, int]] = set()
        for off in offsets:
            if off not in seen:
                seen.add(off)
                unique_offsets.append(off)
        return unique_offsets

    def _maybe_prepare_probe_targets(self, s: AgentState) -> None:
        if not self._should_use_probes or s.hub_probe_initialized or s.home_base_row < 0:
            return
        if not self._hub_probe_offsets:
            s.hub_probe_initialized = True
            return
        rotation = s.agent_id % max(1, len(self._hub_probe_offsets))
        ordered = self._hub_probe_offsets[rotation:] + self._hub_probe_offsets[:rotation]
        for dr, dc in ordered:
            target = (s.home_base_row + dr, s.home_base_col + dc)
            if self._is_valid_position(*target):
                s.hub_probe_targets.append(target)
        s.hub_probe_initialized = True

    def _clear_probe_state(self, s: AgentState) -> None:
        s.hub_probe_targets.clear()
        s.current_probe_target = None

    def _all_core_extractors_known(self) -> bool:
        return all(self.extractor_memory.get_by_type(r) for r in self._gather_resources)

    def _ensure_probe_target(self, s: AgentState) -> Optional[Tuple[int, int]]:
        if not s.hub_probe_targets:
            return None
        if s.current_probe_target is not None:
            return s.current_probe_target
        while s.hub_probe_targets:
            candidate = s.hub_probe_targets[0]
            if self._is_valid_position(*candidate):
                s.current_probe_target = candidate
                return candidate
            s.hub_probe_targets.popleft()
        return None

    def _plan_probe_action(self, s: AgentState, focus_resource: Optional[str]) -> Optional[int]:
        if not self._should_use_probes or focus_resource is None:
            return None
        if self.extractor_memory.get_by_type(focus_resource):
            self._clear_probe_state(s)
            return None
        if self._all_core_extractors_known():
            self._clear_probe_state(s)
            return None
        self._maybe_prepare_probe_targets(s)
        target = self._ensure_probe_target(s)
        if target is None:
            return None
        if (s.agent_row, s.agent_col) == target:
            s.hub_probe_targets.popleft()
            s.current_probe_target = None
            return self._action_lookup.get("noop", 0)

        res = self.navigator.navigate_to(
            start=(s.agent_row, s.agent_col),
            target=target,
            occupancy_map=s.occupancy_map,
            optimistic=True,
            use_astar=C.USE_ASTAR,
            astar_threshold=C.ASTAR_THRESHOLD,
        )

        if res.is_adjacent and res.next_step is None:
            tr, tc = target
            return self._step_toward(tr - s.agent_row, tc - s.agent_col)
        if res.next_step:
            nr, nc = res.next_step
            return self._step_toward(nr - s.agent_row, nc - s.agent_col)

        # Failed to path; drop this probe and try others next time
        if s.hub_probe_targets:
            s.hub_probe_targets.popleft()
        s.current_probe_target = None
        return None

    def _release_resource_focus(self, resource: Optional[str]) -> None:
        if not resource:
            return
        count = self._resource_focus_counts.get(resource, 0)
        if count > 0:
            self._resource_focus_counts[resource] = count - 1

    def _acquire_resource_focus(self, resource: Optional[str]) -> None:
        if not resource:
            return
        self._resource_focus_counts[resource] = self._resource_focus_counts.get(resource, 0) + 1

    def _update_recharge_progress(self, s: AgentState) -> None:
        if s.current_phase == GamePhase.RECHARGE:
            if s.recharge_last_energy < 0:
                s.recharge_last_energy = s.energy
            if s.energy_delta > 0:
                s.recharge_total_gained += s.energy_delta
                s.recharge_ticks_without_gain = 0
                s.recharge_last_energy = s.energy
            elif s.last_attempt_was_use:
                s.recharge_ticks_without_gain += 1
            else:
                s.recharge_last_energy = s.energy
        else:
            s.recharge_last_energy = -1
            s.recharge_ticks_without_gain = 0
            s.recharge_total_gained = 0

    def _can_focus_resource(self, resource: Optional[str]) -> bool:
        if not resource:
            return True
        limit = self._resource_focus_limits.get(resource)
        if limit is None:
            return True
        return self._resource_focus_counts.get(resource, 0) < limit

    def _assign_assembler_slot(self, agent_id: int) -> Optional[int]:
        if agent_id in self._assembler_slot_for_agent:
            return self._assembler_slot_for_agent[agent_id]
        occupied = set(self._assembler_slot_for_agent.values())
        for idx, _ in enumerate(self._assembler_offsets):
            if idx not in occupied:
                self._assembler_slot_for_agent[agent_id] = idx
                return idx
        return None

    def _release_assembler_slot(self, agent_id: int) -> None:
        self._assembler_slot_for_agent.pop(agent_id, None)

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
        prev_energy = s.energy
        s.carbon = self._read_int_feature(obs, FeatureNames.INV_CARBON)
        s.oxygen = self._read_int_feature(obs, FeatureNames.INV_OXYGEN)
        s.germanium = self._read_int_feature(obs, FeatureNames.INV_GERMANIUM)
        s.silicon = self._read_int_feature(obs, FeatureNames.INV_SILICON)
        s.energy = self._read_int_feature(obs, FeatureNames.INV_ENERGY)
        s.energy_delta = s.energy - prev_energy
        s.last_energy = prev_energy
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
        if state is None:
            # This should only happen if called incorrectly - state should be managed by StatefulAgentPolicy
            s = self.agent_state()
        else:
            s = state
        if s.active_resource_target:
            self._release_resource_focus(s.active_resource_target)
            s.active_resource_target = None
        self._target_assignments.pop(s.agent_id, None)
        s.step_count += 1

        self._update_inventory(obs, s)
        self._update_agent_position(s)
        self._update_rewards(obs, s)
        self._update_recharge_progress(s)

        # Home
        if s.home_base_row == -1 and s.agent_row >= 0:
            s.home_base_row, s.home_base_col = s.agent_row, s.agent_col
            logger.info(f"[Init] Home base: ({s.home_base_row},{s.home_base_col})")
        self._maybe_prepare_probe_targets(s)

        self._mark_cell(s, s.agent_row, s.agent_col, C.OCC_FREE)
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
            pos = (s.agent_row, s.agent_col)
            s.visited_cells.add(pos)
            s.visit_counts[pos] = s.visit_counts.get(pos, 0) + 1

        # Deposit detection
        if (s.last_heart > 0 and s.heart == 0) or (s.current_phase == GamePhase.DEPOSIT_HEART and s.last_reward > 0):
            s.hearts_assembled += 1
            s.wait_counter = 0
            s.current_phase = GamePhase.GATHER_GERMANIUM
            s.just_deposited = True
            logger.info(f"[Deposit] Heart deposited -> total={s.hearts_assembled}")
            self._assembly_signal = {"active": False, "requester": None, "position": None}
            self._assembly_signal_participants.clear()
            self._assembler_slot_for_agent.clear()
            self._target_assignments.clear()
            self._release_assembler_slot(s.agent_id)

        # Phase selection via controller
        old_phase = s.current_phase
        s.current_phase = self._determine_phase(s, obs)
        if s.current_phase != old_phase:
            # Track phase history (keep last 5 phases for returning after interrupts)
            s.phase_history.append(old_phase)
            if len(s.phase_history) > 5:
                s.phase_history.pop(0)

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

        if self._target_assignments.get(s.agent_id) is not None:
            self._acquire_resource_focus(s.active_resource_target)
        s.last_action_idx = action_idx
        s.prev_pos = (s.agent_row, s.agent_col)
        s.last_heart = s.heart
        if s.current_phase == GamePhase.ASSEMBLE_HEART:
            self._assembly_signal_participants.add(s.agent_id)
        else:
            self._assembly_signal_participants.discard(s.agent_id)
        self._log_agent_status(s)
        return dtype_actions.type(action_idx), s

    # ------------------------ Phase determination ------------------------
    def _determine_phase(self, s: AgentState, obs: MettaGridObservation) -> GamePhase:
        ctx = Context(obs=obs, env=self._env, step=s.step_count)
        # Make available to guards either via ctx or env
        ctx.policy_impl = self
        if not hasattr(self._env, "policy_impl"):
            self._env.policy_impl = self
        assembler_known = have_assembler_discovered(s, ctx)
        materials_ready = assembler_known and has_all_materials(s, ctx)

        if self._assembly_signal["active"] and not assembler_known:
            self._assembly_signal = {"active": False, "requester": None, "position": None}
            self._assembly_signal_participants.clear()
            self._assembler_slot_for_agent.clear()

        if s.is_leader:
            if assembler_known and materials_ready:
                if not self._assembly_signal["active"]:
                    self._assembly_signal = {
                        "active": True,
                        "requester": s.agent_id,
                        "position": self._station_positions.get("assembler"),
                    }
                    self._assembly_signal_participants.clear()
                    self._assembler_slot_for_agent.clear()
            elif self._assembly_signal["active"] and self._assembly_signal.get("requester") == s.agent_id:
                self._assembly_signal = {"active": False, "requester": None, "position": None}
                self._assembly_signal_participants.clear()
                self._assembler_slot_for_agent.clear()

        proposed = self.phase_controller.maybe_transition(s, ctx, logger)

        if self._assembly_signal["active"] and assembler_known:
            if self._assembly_signal["position"] is None:
                self._assembly_signal["position"] = self._station_positions.get("assembler")
            if materials_ready:
                proposed = GamePhase.ASSEMBLE_HEART
                s.active_resource_target = "assembler"

        if proposed in self._gather_phases:
            deficits = self._resource_deficits(s)
            focus = self._choose_resource_focus(s, proposed, deficits)
            s.active_resource_target = focus
            if focus is not None:
                desired = self._resource_to_phase.get(focus, proposed)
                if desired != proposed:
                    logger.info(
                        "[Agent %s] overriding phase %s → %s (focus=%s, deficits=%s)",
                        s.agent_id,
                        proposed.name,
                        desired.name,
                        focus,
                        deficits,
                    )
                proposed = desired
        elif proposed == GamePhase.ASSEMBLE_HEART:
            slot = self._assign_assembler_slot(s.agent_id)
            if slot is None:
                logger.info("[Agent %s] no assembler slot available; continuing exploration", s.agent_id)
                self._release_resource_focus("assembler")
                self._release_assembler_slot(s.agent_id)
                s.active_resource_target = None
                return GamePhase.EXPLORE
            s.active_resource_target = "assembler"
        else:
            s.active_resource_target = None
            self._release_assembler_slot(s.agent_id)

        return proposed

    # ---------------------- Phase → concrete action ----------------------
    def _execute_phase(self, s: AgentState) -> int:
        if s.current_phase == GamePhase.EXPLORE:
            s.last_attempt_was_use = False
            plan = self._plan_to_frontier_action(s)
            return plan if plan is not None else self._explore_simple(s)

        if s.current_phase == GamePhase.UNCLIP_STATION:
            return self._do_unclip(s)

        if s.current_phase == GamePhase.CRAFT_DECODER:
            return self._do_craft_decoder(s)

        station = self._phase_to_station.get(s.current_phase)
        target: Optional[Tuple[int, int]] = None

        if s.current_phase == GamePhase.ASSEMBLE_HEART:
            station = "assembler"
            asm = self._station_positions.get("assembler")
            if asm is None:
                s.last_attempt_was_use = False
                plan = self._plan_to_frontier_action(s)
                return plan if plan is not None else self._explore_simple(s)
            slot = self._assembler_slot_for_agent.get(s.agent_id)
            if slot is None:
                slot = self._assign_assembler_slot(s.agent_id)
            if slot is None:
                logger.info("[Agent %s] assembler full, resuming exploration", s.agent_id)
                s.last_attempt_was_use = False
                return self._action_lookup.get("noop", 0)
            offset_r, offset_c = self._assembler_offsets[slot]
            approach = (asm[0] + offset_r, asm[1] + offset_c)
            if (s.agent_row, s.agent_col) != approach:
                target = approach
            else:
                target = asm
            self._target_assignments[s.agent_id] = target
            s.explore_goal = None
        elif s.current_phase in self._gather_phases and s.agent_row != -1:
            deficits = self._resource_deficits(s)
            focus = s.active_resource_target or self._phase_to_resource.get(s.current_phase)
            focus, target = self._resolve_gather_target(s, focus, deficits)
            if target is None or focus is None:
                probe_action = self._plan_probe_action(s, focus)
                if probe_action is not None:
                    s.last_attempt_was_use = False
                    return probe_action
                logger.info(
                    "[Agent %s] %s: no extractor available (focus=%s, deficits=%s) -> exploring",
                    s.agent_id,
                    s.current_phase.name,
                    focus,
                    deficits,
                )
                s.last_attempt_was_use = False
                plan = self._plan_to_frontier_action(s)
                return plan if plan is not None else self._explore_simple(s)
            station = self._resource_to_station.get(focus, station)
            logger.info(
                "[Agent %s] %s targeting %s extractor at %s",
                s.agent_id,
                s.current_phase.name,
                focus,
                target,
            )
            s.explore_goal = None
        elif s.current_phase == GamePhase.RECHARGE and s.agent_row != -1:
            target = self._find_best_extractor_for_phase(GamePhase.RECHARGE, s)
            if target is None:
                logger.info("[Agent %s] recharge: no charger known yet", s.agent_id)
                s.last_attempt_was_use = False
                plan = self._plan_to_frontier_action(s)
                return plan if plan is not None else self._explore_simple(s)
            self._target_assignments[s.agent_id] = target
            s.explore_goal = None
        else:
            target = self._station_positions.get(station) if station else None
            if station == "assembler" and target is not None:
                self._target_assignments[s.agent_id] = target
            if target is not None:
                s.explore_goal = None

        if station is None:
            return self._action_lookup.get("noop", 0)

        # Glyph selection (after station possibly updated)
        need_glyph = self._phase_to_glyph.get(s.current_phase, self._station_to_glyph.get(station, "default"))
        if s.current_glyph != need_glyph:
            s.current_glyph = need_glyph
            s.wait_counter = 0
            gid = self._glyph_name_to_id.get(need_glyph, 0)
            return self._action_lookup.get(f"change_glyph_{gid}", self._action_lookup.get("noop", 0))

        if target is None:
            s.last_attempt_was_use = False
            plan = self._plan_to_frontier_action(s)
            return plan if plan is not None else self._explore_simple(s)

        # Navigate
        res = self.navigator.navigate_to(
            start=(s.agent_row, s.agent_col),
            target=target,
            occupancy_map=s.occupancy_map,
            optimistic=True,
            use_astar=C.USE_ASTAR,
            astar_threshold=C.ASTAR_THRESHOLD,
        )

        if res.is_adjacent:
            e = self.extractor_memory.get_at_position(target)
            if e and s.wait_target == target:
                rem = self.cooldown_remaining(e, s.step_count)
                resource_type = e.resource_type
                patience_limit = C.PATIENCE_STEPS
                allow_alternatives = True
                if resource_type:
                    allow_alternatives = len(self.extractor_memory.get_by_type(resource_type)) > 1
                if resource_type in {"silicon", "carbon"}:
                    patience_limit = max(C.PATIENCE_STEPS * 5, 40)
                if rem > self.hyperparams.wait_if_cooldown_leq:  # hyperparam respected
                    if allow_alternatives and self._exists_viable_alternative(e.resource_type, s, C.ALT_ROTATE_RADIUS):
                        s.last_attempt_was_use = False
                        logger.debug(f"[Wait→Rotate] cooldown~{rem}")
                        return self._navigate_to_best_alternative(e.resource_type, s)
                    if s.waiting_since_step < 0:
                        s.waiting_since_step = s.step_count
                    if s.step_count - s.waiting_since_step <= patience_limit:
                        s.last_attempt_was_use = False
                        return self._action_lookup.get("noop", 0)
                    if allow_alternatives:
                        return self._navigate_to_best_alternative(e.resource_type, s)
                    s.waiting_since_step = s.step_count
                    s.last_attempt_was_use = False
                    return self._action_lookup.get("noop", 0)
                s.wait_target = None
                s.waiting_since_step = -1

            tr, tc = target
            s.last_attempt_was_use = True
            a = self._step_toward(tr - s.agent_row, tc - s.agent_col)
            return a

        if res.next_step:
            nr, nc = res.next_step
            s.last_attempt_was_use = False
            return self._step_toward(nr - s.agent_row, nc - s.agent_col)

        # Stuck → explore
        s.last_attempt_was_use = False
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
                s.last_attempt_was_use = False
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
            occupancy_map=s.occupancy_map,
            optimistic=True,
            use_astar=C.USE_ASTAR,
            astar_threshold=C.ASTAR_THRESHOLD,
        )
        if res.is_adjacent:
            tr, tc = tgt
            s.last_attempt_was_use = True
            return self._step_toward(tr - s.agent_row, tc - s.agent_col)
        if res.next_step:
            nr, nc = res.next_step
            s.last_attempt_was_use = False
            return self._step_toward(nr - s.agent_row, nc - s.agent_col)

        s.last_attempt_was_use = False
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
            s.last_attempt_was_use = False
            plan = self._plan_to_frontier_action(s)
            return plan if plan is not None else self._explore_simple(s)

        res = self.navigator.navigate_to(
            start=(s.agent_row, s.agent_col),
            target=asm,
            occupancy_map=s.occupancy_map,
            optimistic=True,
            use_astar=C.USE_ASTAR,
            astar_threshold=C.ASTAR_THRESHOLD,
        )
        if res.is_adjacent:
            tr, tc = asm
            s.last_attempt_was_use = True
            return self._step_toward(tr - s.agent_row, tc - s.agent_col)
        if res.next_step:
            nr, nc = res.next_step
            s.last_attempt_was_use = False
            return self._step_toward(nr - s.agent_row, nc - s.agent_col)

        s.last_attempt_was_use = False
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
        deficits = self._resource_deficits(s)
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
                must_wait = r in {"silicon", "carbon"} and deficits.get(r, 0) > 0 and len(xs) == 1
                if should_wait or must_wait:
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
            (nr, nc)
            for nr, nc in self._neighbors4(tr, tc)
            if s.occupancy_map[nr][nc] == C.OCC_FREE and (nr, nc) != (sr, sc)
        ]
        if not neighbors:
            return None
        neighbors.sort(key=lambda p: (s.visit_counts.get(p, 0), abs(p[0] - sr) + abs(p[1] - sc)))
        for goal in neighbors:
            step = self._bfs_next_step_occ(s, (sr, sc), goal)
            if step is not None:
                return self._step_toward(step[0] - sr, step[1] - sc)
        return None

    def _choose_frontier(self, s: AgentState) -> Optional[Tuple[int, int]]:
        if s.agent_row < 0:
            return None
        start = (s.agent_row, s.agent_col)
        fronts = set(self._compute_frontiers(s))
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
                    if (nr, nc) in seen or s.occupancy_map[nr][nc] != C.OCC_FREE:
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
        return self._choose_frontier_bfs(s, start, fronts)

    def _choose_frontier_bfs(
        self, s: AgentState, start: Tuple[int, int], fronts: Set[Tuple[int, int]]
    ) -> Optional[Tuple[int, int]]:
        q, seen = deque([start]), {start}
        while q:
            r, c = q.popleft()
            neighbors = sorted(
                self._neighbors4(r, c),
                key=lambda pos: (s.visit_counts.get(pos, 0), abs(pos[0] - start[0]) + abs(pos[1] - start[1])),
            )
            for nr, nc in neighbors:
                if (nr, nc) in fronts:
                    return (nr, nc)
            for nr, nc in neighbors:
                if (nr, nc) in seen or s.occupancy_map[nr][nc] != C.OCC_FREE:
                    continue
                seen.add((nr, nc))
                q.append((nr, nc))
        return None

    def _compute_frontiers(self, s: AgentState) -> List[Tuple[int, int]]:
        res = []
        for r in range(self._map_h):
            for c in range(self._map_w):
                if s.occupancy_map[r][c] != C.OCC_UNKNOWN:
                    continue
                for nr, nc in self._neighbors4(r, c):
                    if s.occupancy_map[nr][nc] == C.OCC_FREE:
                        res.append((r, c))
                        break
        return res

    # ------------------------------ Movement -----------------------------
    def _bfs_next_step_occ(
        self, s: AgentState, start: Tuple[int, int], goal: Tuple[int, int]
    ) -> Optional[Tuple[int, int]]:
        return self._bfs_next_step(s, start, goal, optimistic=False)

    def _bfs_next_step_optimistic(
        self, s: AgentState, start: Tuple[int, int], goal: Tuple[int, int]
    ) -> Optional[Tuple[int, int]]:
        return self._bfs_next_step(s, start, goal, optimistic=True)

    def _bfs_next_step(
        self, s: AgentState, start: Tuple[int, int], goal: Tuple[int, int], optimistic: bool
    ) -> Optional[Tuple[int, int]]:
        if start == goal:
            return start
        q, parent = deque([start]), {start: None}
        while q:
            r, c = q.popleft()
            neighbors = sorted(
                self._neighbors4(r, c),
                key=lambda pos: (s.visit_counts.get(pos, 0), abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])),
            )
            for nr, nc in neighbors:
                if (nr, nc) in parent or not self._is_cell_passable(s, nr, nc, optimistic):
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

    def _is_cell_passable(self, s: AgentState, r: int, c: int, optimistic: bool = False) -> bool:
        cell = s.occupancy_map[r][c]
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
            occupancy_map=s.occupancy_map,
            optimistic=True,
            use_astar=C.USE_ASTAR,
            astar_threshold=C.ASTAR_THRESHOLD,
        )
        if res.next_step:
            nr, nc = res.next_step
            s.last_attempt_was_use = False
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
                self._mark_cell(s, r, c, C.OCC_OBSTACLE)
                continue

            if tid in self._type_id_to_station:
                station = self._type_id_to_station[tid]
                self._mark_cell(s, r, c, C.OCC_OBSTACLE)

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

            # Mark all non-wall, non-station cells as FREE (including other agents)
            # This allows agents to pathfind through each other's positions
            self._mark_cell(s, r, c, C.OCC_FREE)

        self._mark_cell(s, s.agent_row, s.agent_col, C.OCC_FREE)

    # ----------------------------- Bookkeeping ---------------------------
    def _update_wall_knowledge(self, s: AgentState) -> None:
        if not s.prev_pos or s.last_action_idx not in self._MOVE_SET:
            return
        cur = (s.agent_row, s.agent_col)
        if cur != s.prev_pos:
            self._mark_cell(s, *cur, C.OCC_FREE)
            return
        if s.last_attempt_was_use:
            s.last_attempt_was_use = False
            return

        # In multi-agent scenarios, disable wall poisoning entirely
        # Agents can block each other temporarily, and by the time we check the occupancy map,
        # the blocking agent may have moved, causing us to incorrectly poison that cell
        if self._env.c_env.num_agents > 1:
            return

        dr, dc = self._action_to_dir(s.last_action_idx)
        if dr is None:
            return
        wr, wc = s.prev_pos[0] + dr, s.prev_pos[1] + dc
        if not self._is_valid_position(wr, wc):
            return

        # Mark as obstacle (wall)
        s.occupancy_map[wr][wc] = C.OCC_OBSTACLE

    def _update_agent_position(self, s: AgentState) -> None:
        try:
            for _id, obj in self._env.c_env.grid_objects().items():
                if obj.get("agent_id") == s.agent_id:
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
        prev = self._prev_inventory.get(s.agent_id)
        if prev is None:
            self._prev_inventory[s.agent_id] = cur.copy()
            return {}
        changes = {k: cur[k] - prev.get(k, cur[k]) for k in cur if cur[k] != prev.get(k, cur[k])}
        self._prev_inventory[s.agent_id] = cur.copy()
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

    def _mark_cell(self, s: AgentState, r: int, c: int, cell_type: int) -> None:
        if self._is_valid_position(r, c):
            s.occupancy_map[r][c] = cell_type

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
        cur = (s.agent_row, s.agent_col)
        history = self._recent_positions.setdefault(s.agent_id, deque(maxlen=self._max_recent_positions))
        if not history or history[-1] != cur:
            history.append(cur)
        history_list = list(history)
        history_len = len(history_list)
        visit_counts = s.visit_counts

        row_parity = s.agent_row % 2
        preferred = self._MOVE_E if row_parity == 0 else self._MOVE_W
        dr, dc = self._action_to_dir(preferred)
        nr, nc = s.agent_row + (dr or 0), s.agent_col + (dc or 0)
        if preferred in self._MOVE_SET and self._is_valid_position(nr, nc) and self._is_cell_passable(s, nr, nc):
            return preferred

        down_r, down_c = s.agent_row + 1, s.agent_col
        if self._is_valid_position(down_r, down_c) and self._is_cell_passable(s, down_r, down_c):
            return self._MOVE_S if self._MOVE_S != -1 else self._action_lookup.get("noop", 0)

        # Choose any in-bounds, passable, not-recent direction
        options = [(self._MOVE_N, (-1, 0)), (self._MOVE_S, (1, 0)), (self._MOVE_E, (0, 1)), (self._MOVE_W, (0, -1))]
        best, best_score = None, (-float("inf"), -1)
        for a, (dr, dc) in options:
            nr, nc = s.agent_row + dr, s.agent_col + dc
            if not self._is_valid_position(nr, nc):
                continue
            if not self._is_cell_passable(s, nr, nc):
                continue
            pos = (nr, nc)
            if pos in history_list:
                recency_score = history_len - history_list.index(pos)
            else:
                recency_score = history_len + 1
            visit_score = -visit_counts.get(pos, 0)
            score = (visit_score, recency_score)
            if score > best_score:
                best_score, best = score, a
        return best if best is not None else (preferred if preferred != -1 else self._action_lookup.get("noop", 0))


class ScriptedAgentPolicy(Policy):
    def __init__(self, env: MettaGridEnv | None = None, device=None, hyperparams: Hyperparameters | None = None):
        self._env = env
        self._hyperparams = hyperparams
        self._impl = ScriptedAgentPolicyImpl(env, hyperparams=hyperparams) if env is not None else None
        self._agent_policies: Dict[int, AgentPolicy] = {}  # Cache per-agent policies

    def reset(self, obs, info):
        if self._env is None and "env" in info:
            self._env = info["env"]
        if self._impl is None:
            if self._env is None:
                raise RuntimeError("ScriptedAgentPolicy needs env - provide during __init__ or via info['env']")
            self._impl = ScriptedAgentPolicyImpl(self._env, hyperparams=self._hyperparams)
        # Clear cached agent policies on reset
        self._agent_policies = {}

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        if self._impl is None:
            raise RuntimeError("Policy not initialized - call reset() first")
        # Cache and reuse agent policies to maintain state across calls
        if agent_id not in self._agent_policies:
            self._agent_policies[agent_id] = StatefulAgentPolicy(self._impl, agent_id)
        return self._agent_policies[agent_id]
