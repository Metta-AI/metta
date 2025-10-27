"""Scripted agent that can detect and unclip clipped stations.

Extends the outpost scripted agent with:
- Detect 'clipped' observation on stations
- Craft a decoder at the assembler when needed
- Move to clipped station and attempt to unclip
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

from .scripted_agent_outpost import (
    MettaGridEnv,
    MettaGridObservation,
    StatefulAgentPolicy,
)
from .scripted_agent_outpost import (
    ScriptedAgentPolicy as BasePolicy,
)
from .scripted_agent_outpost import (
    ScriptedAgentPolicyImpl as BaseImpl,
)


class ClippingPhase(Enum):
    CRAFT_DECODER = "craft_decoder"
    UNCLIP_STATION = "unclip_station"


@dataclass
class ClippingState:
    target_clipped_pos: Optional[Tuple[int, int]] = None
    tried_unclip: bool = False
    tried_craft_decoder: bool = False


class ScriptedAgentClippingImpl(BaseImpl):
    """Adds unclipping behavior on top of the outpost policy implementation."""

    def __init__(self, env: MettaGridEnv):
        super().__init__(env)
        # Feature id for 'clipped' flag on stations
        self._clipped_fid: Optional[int] = self._feature_name_to_id.get("clipped")
        # Track which stations are currently clipped
        self._clipped_positions: Dict[Tuple[int, int], bool] = {}
        # Per-agent unclipping state (we run single-agent in TF tests)
        self._clip_state = ClippingState()

    # ---------- Observation → Knowledge (override to add clipped detection) ----------
    def _discover_stations_from_observation(self, obs: MettaGridObservation, state):  # type: ignore[override]
        # First pass: mark which local coords are clipped
        clipped_local: set[Tuple[int, int]] = set()
        if self._clipped_fid is not None and state.agent_row >= 0:
            for tok in obs:
                if self._to_int(tok[1]) == self._clipped_fid and self._to_int(tok[2]) == 1:
                    packed = self._to_int(tok[0])
                    obs_r = packed >> 4
                    obs_c = packed & 0x0F
                    clipped_local.add((obs_r, obs_c))

        # Call base to record stations and walls
        super()._discover_stations_from_observation(obs, state)  # type: ignore[attr-defined]

        # Second pass: associate 'clipped' with known station positions
        if state.agent_row >= 0:
            for obs_r, obs_c in clipped_local:
                map_r = obs_r - self.OBS_HEIGHT_RADIUS + state.agent_row
                map_c = obs_c - self.OBS_WIDTH_RADIUS + state.agent_col
                if self._is_valid_position(map_r, map_c):
                    self._clipped_positions[(map_r, map_c)] = True

        # Clear entries no longer visible if we later find them unclipped (handled when used)

    # ---------- Phase selection (augment) ----------
    def _determine_phase(self, state):  # type: ignore[override]
        # If we have a known clipped station, prioritize unclipping workflow
        clipped_target = self._choose_clipped_target(state)
        if clipped_target is not None:
            self._clip_state.target_clipped_pos = clipped_target
            # Craft decoder first once if we have carbon, then try unclipping
            if not self._clip_state.tried_craft_decoder and state.carbon >= 1:
                return ClippingPhase.CRAFT_DECODER
            return ClippingPhase.UNCLIP_STATION
            # Otherwise, fall through to base policy which will gather carbon

        # Default behavior
        return super()._determine_phase(state)

    # ---------- Execution (augment) ----------
    def _execute_phase(self, state) -> int:  # type: ignore[override]
        # Handle unclipping phases
        if state.current_phase == ClippingPhase.CRAFT_DECODER:
            # Mark that we've attempted crafting so we move on to unclipping next
            self._clip_state.tried_craft_decoder = True
            # Go to assembler, switch to 'gear' glyph, and use it to convert 1 carbon -> decoder
            assembler_pos = self._station_positions.get("assembler")
            if assembler_pos is None:
                # Explore to find assembler
                self._last_attempt_was_use = False
                plan = self._plan_to_frontier_action(state)
                return plan if plan is not None else self._explore_simple(state)

            # Ensure gear glyph
            if state.current_glyph != "gear":
                glyph_id = self._glyph_name_to_id.get("gear", 0)
                return self._action_lookup.get(f"change_glyph_{glyph_id}", self._action_lookup.get("noop", 0))

            # Navigate to assembler and attempt use
            return self._navigate_and_use(state, assembler_pos)

        if state.current_phase == ClippingPhase.UNCLIP_STATION:
            target = self._clip_state.target_clipped_pos
            if not target:
                return self._action_lookup.get("noop", 0)
            # Attempt to use the clipped station (no specific glyph needed for unclip)
            action = self._navigate_and_use(state, target)
            # If we reach adjacency and use, the station should unclip and inventory may change
            return action

        # Fallback to base
        return super()._execute_phase(state)

    # ---------- Helpers ----------
    def _read_int_feature(self, obs: Optional[MettaGridObservation] = None, state=None, feat_name: str = "") -> int:
        # Expose base helper for inventory reading without requiring obs
        if obs is not None:
            return super()._read_int_feature(obs, feat_name)
        # When obs is None, read from cached mapping via last obs step if available -> fallback to state vars
        # We only need 'inv:decoder' and carbon is already tracked on state; decoder may not be on state, so return 0
        return 0

    def _choose_clipped_target(self, state) -> Optional[Tuple[int, int]]:
        # Pick the nearest known clipped station to current position
        if state.agent_row < 0 or not self._clipped_positions:
            return None
        sr, sc = state.agent_row, state.agent_col
        best, best_d = None, 10**9
        for (r, c), is_clipped in list(self._clipped_positions.items()):
            if not is_clipped:
                continue
            d = abs(r - sr) + abs(c - sc)
            if d < best_d:
                best, best_d = (r, c), d
        return best

    def _navigate_and_use(self, state, target_pos: Tuple[int, int]) -> int:
        # Use navigator to go to target; when adjacent, step into it to perform use
        start_pos = (state.agent_row, state.agent_col)
        result = self.navigator.navigate_to(
            start=start_pos,
            target=target_pos,
            occupancy_map=self._occ,
            optimistic=True,
            use_astar=self.hyperparams.use_astar,
            astar_threshold=self.hyperparams.astar_threshold,
        )
        if result.is_adjacent:
            tr, tc = target_pos
            dr, dc = tr - state.agent_row, tc - state.agent_col
            self._last_attempt_was_use = True
            return self._step_toward(dr, dc)
        if result.next_step:
            nr, nc = result.next_step
            dr, dc = nr - state.agent_row, nc - state.agent_col
            self._last_attempt_was_use = False
            return self._step_toward(dr, dc)
        # If stuck, explore
        self._last_attempt_was_use = False
        plan = self._plan_to_frontier_action(state)
        return plan if plan is not None else self._explore_simple(state)


class ScriptedAgentPolicy(BasePolicy):
    """Public policy wrapper for clipping-aware scripted agent."""

    def __init__(self, env: MettaGridEnv | None = None, device=None):
        self._env = env
        if env is not None:
            self._impl = ScriptedAgentClippingImpl(env)
        else:
            self._impl = None

    def reset(self, obs, info):
        if self._env is None and "env" in info:
            self._env = info["env"]
        if self._impl is None:
            if self._env is None:
                raise RuntimeError("ScriptedAgentPolicy needs env - provide during __init__ or in info['env']")
            self._impl = ScriptedAgentClippingImpl(self._env)

    def agent_policy(self, agent_id: int) -> StatefulAgentPolicy:
        if self._impl is None:
            raise RuntimeError("Policy not initialized - call reset() first")
        return StatefulAgentPolicy(self._impl, agent_id)
