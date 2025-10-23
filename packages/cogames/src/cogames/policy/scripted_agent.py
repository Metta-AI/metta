"""Scripted agent policy for CoGames training facility missions with detailed logging.

Key points:
- Inventory comes only from observation tokens; if a token is absent, value is 0.
- Heart is considered PRESENT (1) only if the FIRST 'inv:heart' token's first field equals 85 (0x55).
- No swapping / heuristics — just read what the obs says.
- Simple BFS to known targets; we discover absolute agent position from env.c_env.grid_objects().
- Border detection: if we tried to move but position did not change (and last tick wasn’t a 'use' step),
  we schedule a corrective turn this tick to avoid getting stuck on walls/borders.
- On deposit detection (reward>0 while depositing OR heart transitions 1->0), we immediately re-target gather.

You don't get an observation for an inventory that you don't have.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List, Tuple

import logging
import os
from collections import deque

from cogames.policy.interfaces import AgentPolicy, Policy, StatefulAgentPolicy, StatefulPolicyImpl
from mettagrid import MettaGridAction, MettaGridEnv, MettaGridObservation, dtype_actions

logger = logging.getLogger("cogames.policy.scripted_agent")


# ===============================
# Enums & Data
# ===============================
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
    """Scripted policy with absolute position tracking and detailed logging."""

    # ---- Constants & thresholds ----
    RECHARGE_THRESHOLD = 40
    CARBON_REQ = 20
    OXYGEN_REQ = 20
    SILICON_REQ = 50
    ENERGY_REQ = 20
    HEART_FEATURE_NAME = "inv:heart"
    HEART_SENTINEL_FIRST_FIELD = 85  # 0x55

    def __init__(self, env: MettaGridEnv):
        self._env = env
        self._action_names: List[str] = env.action_names
        self._object_type_names: List[str] = env.object_type_names

        # Lookups
        self._action_lookup: Dict[str, int] = {name: i for i, name in enumerate(self._action_names)}

        obs_features = env.observation_features
        self._feature_name_to_id: Dict[str, int] = {feature.name: feature.id for feature in obs_features.values()}

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

        # Phase → station
        self._phase_to_station: Dict[GamePhase, str] = {
            GamePhase.GATHER_GERMANIUM: "germanium_extractor",
            GamePhase.GATHER_SILICON: "silicon_extractor",
            GamePhase.GATHER_CARBON: "carbon_extractor",
            GamePhase.GATHER_OXYGEN: "oxygen_extractor",
            GamePhase.ASSEMBLE_HEART: "assembler",
            GamePhase.DEPOSIT_HEART: "chest",
            GamePhase.RECHARGE: "charger",
        }

        # World geometry
        self._station_positions = self._get_station_positions()
        self._wall_positions = self._get_wall_positions()
        self._map_height = env.c_env.map_height
        self._map_width = env.c_env.map_width

        # --- Border detection / movement memory ---
        self._prev_pos: Optional[Tuple[int, int]] = None
        self._last_action_idx: Optional[int] = None
        self._last_attempt_was_use: bool = False
        self._override_action_idx: Optional[int] = None
        self._MOVE_N = self._action_lookup.get("move_north", -1)
        self._MOVE_S = self._action_lookup.get("move_south", -1)
        self._MOVE_E = self._action_lookup.get("move_east", -1)
        self._MOVE_W = self._action_lookup.get("move_west", -1)
        self._MOVE_SET = {self._MOVE_N, self._MOVE_S, self._MOVE_E, self._MOVE_W}

        # Trajectory CSV header log (stdout prints each step)
        logger.info("Scripted agent initialized with position tracking")
        logger.info(f"Map size: {self._map_height}x{self._map_width}")
        logger.info(f"Station positions: {self._station_positions}")
        logger.info(f"Found {len(self._wall_positions)} walls")
        logger.info(
            "Inv feature IDs: "
            + str([(k, self._feature_name_to_id.get(k)) for k in ("inv:carbon", "inv:oxygen", "inv:germanium", "inv:silicon", "inv:energy", "inv:heart")])
        )

    # ---------- Environment Introspection ----------
    def _get_station_positions(self) -> Dict[str, Tuple[int, int]]:
        positions: Dict[str, Tuple[int, int]] = {}
        try:
            for _obj_id, obj in self._env.c_env.grid_objects().items():
                if "agent_id" in obj:
                    continue
                tid = obj.get("type_id")
                r, c = obj.get("r"), obj.get("c")
                if tid is None or r is None or c is None:
                    continue
                name = self._object_type_names[tid] if tid < len(self._object_type_names) else None
                if name in self._station_to_glyph:
                    positions[name] = (r, c)
        except Exception as e:
            logger.warning(f"Could not get station positions: {e}")
        return positions

    def _get_wall_positions(self) -> set[Tuple[int, int]]:
        walls: set[Tuple[int, int]] = set()
        try:
            for _obj_id, obj in self._env.c_env.grid_objects().items():
                tid = obj.get("type_id")
                if tid is None:
                    continue
                name = self._object_type_names[tid] if tid < len(self._object_type_names) else None
                if name == "wall":
                    r, c = obj.get("r"), obj.get("c")
                    if r is not None and c is not None:
                        walls.add((r, c))
        except Exception as e:
            logger.warning(f"Could not get wall positions: {e}")
        return walls

    # ---------- Policy Interface ----------
    def agent_state(self) -> AgentState:
        return AgentState()

    def step_with_state(
        self, obs: MettaGridObservation, state: Optional[AgentState]
    ) -> tuple[MettaGridAction, Optional[AgentState]]:
        if state is None:
            state = self.agent_state()
        state.step_count += 1

        # Update from observation
        self._update_inventory(obs, state)
        self._update_agent_position(state)
        self._update_rewards(obs, state)

        # Border detection (based on last tick's move vs current position)
        self._maybe_apply_border_correction(state)

        # Deposit detection:
        #  (a) heart transitioned 1->0, or
        #  (b) while in deposit phase we saw positive last_reward
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

        # # Log (stdout)
        # self._print_step_log(action_idx, state)

        # Bookkeeping for next-tick border detection
        self._last_action_idx = action_idx
        self._prev_pos = (state.agent_row, state.agent_col)

        # Track heart
        state.last_heart = state.heart

        return dtype_actions.type(action_idx), state

    # ---------- Update helpers ----------
    @staticmethod
    def _to_int(x) -> int:
        try:
            return int(x)
        except Exception:
            try:
                return int(x.item())  # numpy scalar
            except Exception:
                return int(x)  # last resort

    def _update_agent_position(self, state: AgentState) -> None:
        """Update agent's absolute position from env.c_env.grid_objects."""
        try:
            for _obj_id, obj in self._env.c_env.grid_objects().items():
                if obj.get("agent_id") == 0:
                    state.agent_row = obj.get("r", -1)
                    state.agent_col = obj.get("c", -1)
                    break
        except Exception as e:
            logger.debug(f"Could not update agent position: {e}")

    def _update_rewards(self, obs: MettaGridObservation, state: AgentState) -> None:
        try:
            last_reward_feat = self._feature_name_to_id.get("global:last_reward")
            if last_reward_feat is None:
                return
            for tok in obs:
                fid = self._to_int(tok[1])
                if fid == last_reward_feat:
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
            fid = self._to_int(tok[1])
            if fid == heart_fid:
                first_idx = i
                break

        if first_idx is None:
            return False

        first_tok = obs[first_idx]
        first_field = self._to_int(first_tok[0])
        return first_field == self.HEART_SENTINEL_FIRST_FIELD

    def _read_int_feature(self, obs: MettaGridObservation, feat_name: str) -> int:
        fid = self._feature_name_to_id.get(feat_name)
        if fid is None:
            return 0
        for tok in obs:
            if self._to_int(tok[1]) == fid:
                return self._to_int(tok[2])
        return 0

    def _update_inventory(self, obs: MettaGridObservation, state: AgentState) -> None:
        """Parse inventory fields from tokens; no swaps, no guesses."""
        state.carbon = self._read_int_feature(obs, "inv:carbon")
        state.oxygen = self._read_int_feature(obs, "inv:oxygen")
        state.germanium = self._read_int_feature(obs, "inv:germanium")
        state.silicon = self._read_int_feature(obs, "inv:silicon")
        state.energy = self._read_int_feature(obs, "inv:energy")

        # Heart: special rule (first token's first field must be 85)
        state.heart = 1 if self._has_heart_from_obs(obs) else 0

    # ---------- Border detection ----------
    def _maybe_apply_border_correction(self, state: AgentState) -> None:
        """
        If last action was a move and our position did not change, assume a border/wall
        and set a corrective action (turn). Skips correction if last tick was a 'use' attempt
        (stepping into a station tile that resolves without movement).
        """
        if self._override_action_idx is not None:
            return  # already scheduled

        if self._last_action_idx not in self._MOVE_SET:
            return
        if self._prev_pos is None:
            return

        stayed = (state.agent_row, state.agent_col) == self._prev_pos
        if not stayed:
            return

        if self._last_attempt_was_use:
            return

        # turn right relative to last move
        right_turn = {
            self._MOVE_N: self._MOVE_E,
            self._MOVE_E: self._MOVE_S,
            self._MOVE_S: self._MOVE_W,
            self._MOVE_W: self._MOVE_N,
        }.get(self._last_action_idx, self._action_lookup.get("noop", 0))

        if right_turn == -1:
            # try opposite, then left, then noop
            opposite = {
                self._MOVE_N: self._MOVE_S,
                self._MOVE_S: self._MOVE_N,
                self._MOVE_E: self._MOVE_W,
                self._MOVE_W: self._MOVE_E,
            }.get(self._last_action_idx, self._action_lookup.get("noop", 0))
            if opposite != -1:
                right_turn = opposite
            else:
                left = {
                    self._MOVE_N: self._MOVE_W,
                    self._MOVE_W: self._MOVE_S,
                    self._MOVE_S: self._MOVE_E,
                    self._MOVE_E: self._MOVE_N,
                }.get(self._last_action_idx, self._action_lookup.get("noop", 0))
                right_turn = left if left != -1 else self._action_lookup.get("noop", 0)

        self._override_action_idx = right_turn
        la = self._last_action_idx
        logger.info(
            f"[Border] Stuck on move; turning. last_action="
            f"{self._action_names[la] if 0 <= la < len(self._action_names) else la}, "
            f"new_action={self._action_names[self._override_action_idx] if 0 <= self._override_action_idx < len(self._action_names) else self._override_action_idx}"
        )

    # ---------- Decision helpers ----------
    def _determine_phase(self, state: AgentState) -> GamePhase:
        if state.heart > 0:
            return GamePhase.DEPOSIT_HEART

        germ_needed = 5 if state.hearts_assembled == 0 else max(2, 5 - state.hearts_assembled)

        if (
            state.germanium >= germ_needed
            and state.carbon >= self.CARBON_REQ
            and state.oxygen >= self.OXYGEN_REQ
            and state.silicon >= self.SILICON_REQ
            and state.energy >= self.ENERGY_REQ
        ):
            return GamePhase.ASSEMBLE_HEART

        if state.energy < self.RECHARGE_THRESHOLD:
            return GamePhase.RECHARGE

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
        # If a border correction was scheduled, do it immediately.
        if self._override_action_idx is not None:
            action_idx = self._override_action_idx
            self._override_action_idx = None
            return action_idx

        station = self._phase_to_station.get(state.current_phase)
        if not station:
            return self._action_lookup.get("noop", 0)

        # After a deposit, if we are still on the chest tile, immediately re-target gather to move away.
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
            action_name = f"change_glyph_{glyph_id}"
            return self._action_lookup.get(action_name, self._action_lookup.get("noop", 0))

        # If we know absolute positions, navigate precisely
        if station in self._station_positions and state.agent_row != -1:
            tr, tc = self._station_positions[station]
            dr, dc = tr - state.agent_row, tc - state.agent_col
            manhattan = abs(dr) + abs(dc)

            # Adjacent to station: attempt to 'step into' it (game resolves use w/o moving)
            if manhattan == 1:
                self._last_attempt_was_use = True
                return self._step_toward(dr, dc)

        # Else pathfind
        act = self._navigate_to_station(station, state)
        self._last_attempt_was_use = False
        return act

    # ---------- Navigation ----------
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

    def _navigate_to_station(self, station_name: str, state: AgentState) -> int:
        if station_name not in self._station_positions:
            return self._action_lookup.get("noop", 0)
        if state.agent_row == -1 or state.agent_col == -1:
            return self._fallback_navigate(state.current_phase)

        start = (state.agent_row, state.agent_col)
        goal = self._station_positions[station_name]
        path = self._bfs_pathfind(start, goal)
        if len(path) > 1:
            next_r, next_c = path[1]
            return self._step_toward(next_r - state.agent_row, next_c - state.agent_col)
        return self._action_lookup.get("noop", 0)

    def _bfs_pathfind(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        if start == goal:
            return [start]

        H, W = self._map_height, self._map_width
        walls = self._wall_positions
        stations = self._station_positions

        q = deque([(start, [start])])
        visited = {start}

        while q:
            (r, c), path = q.popleft()

            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nr, nc = r + dr, c + dc

                # Bounds
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                nxt = (nr, nc)

                # Walls
                if nxt in walls:
                    continue

                # Don't traverse other stations (unless the goal)
                if nxt in stations.values() and nxt != goal:
                    continue

                if nxt in visited:
                    continue
                visited.add(nxt)

                new_path = path + [nxt]
                if nxt == goal:
                    return new_path
                q.append((nxt, new_path))

        return []

    def _fallback_navigate(self, phase: GamePhase) -> int:
        if phase in (GamePhase.GATHER_GERMANIUM, GamePhase.GATHER_SILICON, GamePhase.DEPOSIT_HEART):
            return self._MOVE_S if self._MOVE_S != -1 else self._action_lookup.get("noop", 0)
        if phase in (GamePhase.GATHER_CARBON, GamePhase.GATHER_OXYGEN):
            return self._MOVE_N if self._MOVE_N != -1 else self._action_lookup.get("noop", 0)
        return self._MOVE_S if self._MOVE_S != -1 else self._action_lookup.get("noop", 0)

    # ---------- Logging ----------
    def _print_step_log(self, action_idx: int, state: AgentState) -> None:
        try:
            action_name = self._action_names[action_idx] if 0 <= action_idx < len(self._action_names) else "?"
            target_station = self._phase_to_station.get(state.current_phase, "unknown")
            tr, tc = self._station_positions.get(target_station, (-1, -1))
            if state.agent_row != -1 and tr != -1:
                rel = f"({tr - state.agent_row},{tc - state.agent_col})"
            else:
                rel = "?"
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
