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

import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

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

        # Build type_id -> station name mapping for visual discovery
        self._type_id_to_station: Dict[int, str] = {}
        for type_id, name in enumerate(env.object_type_names):
            if name and name != "wall" and not name.startswith("agent"):
                self._type_id_to_station[type_id] = name

        # Map geometry
        self._map_height = env.c_env.map_height
        self._map_width = env.c_env.map_width

        # Incremental knowledge - NO OMNISCIENCE
        self._station_positions: Dict[str, Tuple[int, int]] = {}  # Discovered through observation
        self._wall_positions: set[Tuple[int, int]] = set()  # Discovered through movement

        # Movement tracking
        self._prev_pos: Optional[Tuple[int, int]] = None
        self._last_action_idx: Optional[int] = None
        self._last_attempt_was_use: bool = False
        self._MOVE_N = self._action_lookup.get("move_north", -1)
        self._MOVE_S = self._action_lookup.get("move_south", -1)
        self._MOVE_E = self._action_lookup.get("move_east", -1)
        self._MOVE_W = self._action_lookup.get("move_west", -1)
        self._MOVE_SET = {self._MOVE_N, self._MOVE_S, self._MOVE_E, self._MOVE_W}

        # Exploration state
        self._exploration_direction: int = self._MOVE_E
        self._stuck_counter: int = 0
        self._recent_positions: List[Tuple[int, int]] = []
        self._max_recent_positions: int = 10

        logger.info("Scripted agent initialized with visual discovery (no omniscience)")
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

    # ---------- Visual Discovery ----------
    def _discover_stations_from_observation(self, obs: MettaGridObservation, state: AgentState) -> None:
        """Discover stations by observing them (type_id feature in observations)."""
        if state.agent_row == -1:
            return

        type_id_feature = self._feature_name_to_id.get("type_id", 0)
        obs_height = 11
        obs_width = 11
        obs_height_radius = obs_height // 2
        obs_width_radius = obs_width // 2

        for tok in obs:
            if tok[0] == 255:  # Sentinel
                break
            if tok[0] == 0x55:  # Inventory token
                continue

            if tok[1] == type_id_feature:
                # Relative coordinates in observation window
                obs_r = int(tok[0] >> 4)
                obs_c = int(tok[0] & 0x0F)
                type_id = int(tok[2])

                # Convert to absolute map coordinates
                map_r = obs_r - obs_height_radius + state.agent_row
                map_c = obs_c - obs_width_radius + state.agent_col

                # Check if this is a station
                if type_id in self._type_id_to_station:
                    station_name = self._type_id_to_station[type_id]
                    pos = (map_r, map_c)

                    if station_name not in self._station_positions:
                        self._station_positions[station_name] = pos
                        logger.info(f"Discovered {station_name} at {pos}")

    def _update_wall_knowledge(self, state: AgentState) -> None:
        """Mark cells as walls when we try to move into them but don't move."""
        if not self._prev_pos or self._last_action_idx not in self._MOVE_SET:
            return

        current_pos = (state.agent_row, state.agent_col)
        if current_pos != self._prev_pos:
            return  # We moved, so no wall

        # We didn't move - mark the target cell as a wall (unless it's a known station)
        dr, dc = self._action_to_dir(self._last_action_idx)
        if dr is None or dc is None:
            return

        wall_r = self._prev_pos[0] + dr
        wall_c = self._prev_pos[1] + dc

        if not (0 <= wall_r < self._map_height and 0 <= wall_c < self._map_width):
            return

        # Don't mark stations as walls
        wall_pos = (wall_r, wall_c)
        is_station = wall_pos in self._station_positions.values()
        if not is_station:
            if wall_pos not in self._wall_positions:
                logger.info(f"Marking wall at {wall_pos} (blocked move {self._last_action_idx})")
            self._wall_positions.add(wall_pos)

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

        # Discover stations visually and walls through movement
        self._discover_stations_from_observation(obs, state)
        self._update_wall_knowledge(state)

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

        # Log (stdout)
        self._print_step_log(action_idx, state)

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
        la_name = self._action_names[la] if 0 <= la < len(self._action_names) else la
        oa = self._override_action_idx
        oa_name = self._action_names[oa] if 0 <= oa < len(self._action_names) else oa
        logger.info(f"[Border] Stuck on move; turning. last_action={la_name}, new_action={oa_name}")

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
        # Border correction disabled - BFS handles navigation
        # if self._override_action_idx is not None:
        #     action_idx = self._override_action_idx
        #     self._override_action_idx = None
        #     return action_idx

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

        # Navigate to station if discovered, otherwise explore
        if station in self._station_positions and state.agent_row != -1:
            tr, tc = self._station_positions[station]
            dr, dc = tr - state.agent_row, tc - state.agent_col

            # Adjacent: step into station
            if abs(dr) + abs(dc) == 1:
                self._last_attempt_was_use = True
                return self._step_toward(dr, dc)

            # Use BFS pathfinding
            path = self._bfs_pathfind((state.agent_row, state.agent_col), (tr, tc))
            if len(path) > 1:
                next_r, next_c = path[1]
                self._last_attempt_was_use = False
                return self._step_toward(next_r - state.agent_row, next_c - state.agent_col)

            # BFS failed, use exploration to navigate around
            return self._explore_simple(state)

        # Station not discovered yet - explore
        self._last_attempt_was_use = False
        return self._explore_simple(state)

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

    # ---------- Exploration ----------
    def _explore_simple(self, state: AgentState) -> int:
        """Boustrophedon (snake) sweep with simple wall-following and loop avoidance."""
        if state.agent_row == -1:
            return self._action_lookup.get("noop", 0)

        # Track position
        current_pos = (state.agent_row, state.agent_col)
        if not self._recent_positions or self._recent_positions[-1] != current_pos:
            self._recent_positions.append(current_pos)
            if len(self._recent_positions) > self._max_recent_positions:
                self._recent_positions.pop(0)

        # Prefer sweeping rows: move east on even rows and west on odd rows,
        # moving south when blocked, with wall-follow fallback.
        row_parity = state.agent_row % 2
        preferred_dir = self._MOVE_E if row_parity == 0 else self._MOVE_W

        # If last move didn't change position, try to move south to next row
        stuck = self._prev_pos and self._last_action_idx in self._MOVE_SET and current_pos == self._prev_pos
        if stuck:
            self._stuck_counter += 1

            # Try to drop down a row if possible
            down_r, down_c = state.agent_row + 1, state.agent_col
            if 0 <= down_r < self._map_height and (down_r, down_c) not in self._wall_positions:
                return self._MOVE_S if self._MOVE_S != -1 else self._action_lookup.get("noop", 0)

            # Otherwise, try alternate horizontal direction
            alt = self._MOVE_W if preferred_dir == self._MOVE_E else self._MOVE_E
            return alt if alt != -1 else self._action_lookup.get("noop", 0)

        # If not stuck, attempt preferred horizontal direction unless known wall
        dr, dc = self._action_to_dir(preferred_dir)
        next_r = state.agent_row + (dr or 0)
        next_c = state.agent_col + (dc or 0)
        if (
            preferred_dir in self._MOVE_SET
            and 0 <= next_r < self._map_height
            and 0 <= next_c < self._map_width
            and (next_r, next_c) not in self._wall_positions
        ):
            return preferred_dir

        # Try moving south when blocked horizontally
        down_r, down_c = state.agent_row + 1, state.agent_col
        if 0 <= down_r < self._map_height and (down_r, down_c) not in self._wall_positions:
            return self._MOVE_S if self._MOVE_S != -1 else self._action_lookup.get("noop", 0)

        # Fallback: choose direction away from recent positions, avoiding known walls
        best_action = self._find_best_exploration_direction(state)
        if best_action is not None:
            return best_action

        return preferred_dir if preferred_dir != -1 else self._action_lookup.get("noop", 0)

    def _find_best_exploration_direction(self, state: AgentState) -> Optional[int]:
        """Find direction to cell not recently visited."""
        directions = [
            (self._MOVE_N, (-1, 0)),
            (self._MOVE_S, (1, 0)),
            (self._MOVE_E, (0, 1)),
            (self._MOVE_W, (0, -1)),
        ]

        best_action = None
        best_score = -1

        for action, (dr, dc) in directions:
            nr = state.agent_row + dr
            nc = state.agent_col + dc

            if not (0 <= nr < self._map_height and 0 <= nc < self._map_width):
                continue

            next_pos = (nr, nc)
            # Avoid known walls
            if next_pos in self._wall_positions:
                continue

            # Score: higher if not in recent positions
            if next_pos not in self._recent_positions:
                score = 10
            else:
                score = self._recent_positions.index(next_pos)

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _action_to_dir(self, action_idx: int) -> Tuple[Optional[int], Optional[int]]:
        if action_idx == self._MOVE_N:
            return (-1, 0)
        if action_idx == self._MOVE_S:
            return (1, 0)
        if action_idx == self._MOVE_E:
            return (0, 1)
        if action_idx == self._MOVE_W:
            return (0, -1)
        return (None, None)

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
