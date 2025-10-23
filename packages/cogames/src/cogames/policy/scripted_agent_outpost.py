"""Exploration-aware scripted agent for Outpost/simple_exploration missions.

Extends the default ScriptedAgent to:
- Treat both depleted and regular germanium extractors as valid early targets
- Prefer regular (outer) germanium after the first heart (kickstart complete)
- Track and avoid stations that appear exhausted (no effect after repeated uses)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from cogames.policy.scripted_agent import (
    AgentState,
    GamePhase,
    ScriptedAgentPolicyImpl,
)
from cogames.policy.interfaces import AgentPolicy, Policy, StatefulAgentPolicy
from mettagrid import MettaGridAction, MettaGridEnv, MettaGridObservation, dtype_actions


@dataclass
class ExplorerState(AgentState):
    # Whether we've completed the initial kickstart (first heart crafted/deposited)
    kickstart_complete: bool = False


class ExplorationAgentPolicyImpl(ScriptedAgentPolicyImpl):
    EXHAUSTION_FAIL_THRESHOLD: int = 2

    def __init__(self, env: MettaGridEnv):
        super().__init__(env)

        # Accept depleted germanium as a valid station mapped to the germanium glyph
        self._station_to_glyph["depleted_germanium_extractor"] = "germanium"

        # Track stations that appear exhausted (coords)
        self._exhausted_positions: Set[Tuple[int, int]] = set()
        self._station_failed_uses: Dict[Tuple[int, int], int] = {}
        self._last_target_station_pos: Optional[Tuple[int, int]] = None

    # ---------- Policy Interface ----------
    def agent_state(self) -> ExplorerState:
        return ExplorerState()

    def step_with_state(
        self, obs: MettaGridObservation, state: Optional[ExplorerState]
    ) -> tuple[MettaGridAction, Optional[ExplorerState]]:
        # Delegate to base to maintain discovery, inventory, and actions
        action, new_state = super().step_with_state(obs, state)

        # Mark kickstart as complete after first heart assembled and deposited
        if new_state is not None and (new_state.hearts_assembled > 0 or new_state.just_deposited):
            new_state.kickstart_complete = True

        # If last step attempted a station use and no significant change occurred, mark as failed
        # This relies on base updating prev_* and current values
        if (
            new_state is not None
            and self._last_target_station_pos is not None
            and self._no_significant_change(new_state)
        ):
            pos = self._last_target_station_pos
            self._station_failed_uses[pos] = self._station_failed_uses.get(pos, 0) + 1
            if self._station_failed_uses[pos] >= self.EXHAUSTION_FAIL_THRESHOLD:
                self._exhausted_positions.add(pos)

        # Reset last target pos each step; it will be set again when we actively step into a station
        self._last_target_station_pos = None
        return action, new_state

    # ---------- Phase execution override ----------
    def _execute_phase(self, state: ExplorerState) -> int:  # type: ignore[override]
        station_candidates = self._candidate_stations_for_phase(state.current_phase, state)

        # Prefer nearest discovered station among candidates that is not exhausted
        target = self._select_best_known_station(station_candidates, state)
        if target is not None:
            tr, tc, station_name = target
            dr, dc = tr - state.agent_row, tc - state.agent_col

            # Switch glyph if needed
            needed_glyph = self._station_to_glyph.get(station_name, "default")
            if state.current_glyph != needed_glyph:
                state.current_glyph = needed_glyph
                glyph_id = self._glyph_name_to_id.get(needed_glyph, 0)
                return self._action_lookup.get(f"change_glyph_{glyph_id}", self._noop())

            # Adjacent: attempt to step in and activate
            if abs(dr) + abs(dc) == 1:
                self._last_target_station_pos = (tr, tc)
                return self._step_toward(dr, dc)

            # Greedy move toward target (same as base)
            if abs(dr) > abs(dc):
                return self._MOVE_S if dr > 0 else self._MOVE_N
            else:
                return self._MOVE_E if dc > 0 else self._MOVE_W

        # Fall back to base exploration if nothing known
        return self._explore_simple(state)

    # ---------- Helpers ----------
    def _candidate_stations_for_phase(self, phase: GamePhase, state: ExplorerState) -> List[str]:
        if phase == GamePhase.GATHER_GERMANIUM:
            # Before kickstart: accept both depleted and regular; after kickstart, prefer regular
            if state.kickstart_complete:
                return ["germanium_extractor", "depleted_germanium_extractor"]
            else:
                return ["depleted_germanium_extractor", "germanium_extractor"]
        if phase == GamePhase.GATHER_SILICON:
            return ["silicon_extractor"]
        if phase == GamePhase.GATHER_CARBON:
            return ["carbon_extractor"]
        if phase == GamePhase.GATHER_OXYGEN:
            return ["oxygen_extractor"]
        if phase == GamePhase.ASSEMBLE_HEART:
            return ["assembler"]
        if phase == GamePhase.DEPOSIT_HEART:
            return ["chest"]
        if phase == GamePhase.RECHARGE:
            return ["charger"]
        return []

    def _select_best_known_station(
        self, station_names: List[str], state: ExplorerState
    ) -> Optional[Tuple[int, int, str]]:
        if state.agent_row == -1:
            return None

        # Gather all discovered stations that match any of the candidate names
        candidates: List[Tuple[int, int, str]] = []
        for name in station_names:
            pos = self._discovered_stations.get(name)
            if pos is None:
                continue
            if pos in self._exhausted_positions:
                continue
            candidates.append((pos[0], pos[1], name))

        if not candidates:
            return None

        # Choose nearest by Manhattan distance
        ar, ac = state.agent_row, state.agent_col
        candidates.sort(key=lambda x: abs(x[0] - ar) + abs(x[1] - ac))
        return candidates[0]

    def _no_significant_change(self, state: ExplorerState) -> bool:
        # Consider significant changes similar to base discovery logic
        if state.germanium > state.prev_germanium:
            return False
        if state.silicon > state.prev_silicon:
            return False
        if state.carbon > state.prev_carbon:
            return False
        if state.oxygen > state.prev_oxygen:
            return False
        if state.energy > state.prev_energy + 5:
            return False
        if state.heart > state.last_heart:
            return False
        return True


class ExplorationAgentPolicy(Policy):
    def __init__(self, env: MettaGridEnv, device=None):
        self._env = env
        self._impl = ExplorationAgentPolicyImpl(env)

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        return StatefulAgentPolicy(self._impl, agent_id)


