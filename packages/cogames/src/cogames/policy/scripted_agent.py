"""Scripted agent policy for CoGames training facility missions with detailed logging.

Key points:
- Inventory comes only from observation tokens; if a token is absent, value is 0.
- Heart is considered PRESENT (1) only if the FIRST 'inv:heart' token's first field equals 85 (0x55).
- No swapping / heuristics — just read what the obs says.
- NO OMNISCIENCE: Stations are discovered visually; walls are learned from failed moves.
- Frontier-based exploration: maintain unknown/free/wall grid; go to nearest frontier (unknown adjacent to free).
- If frontier planning can’t find a step, fall back to a simple sweep. Add a tiny cycle breaker.
- On deposit detection (reward>0 while depositing OR heart transitions 1->0), immediately re-target gather.

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
    """Scripted policy with visual discovery, frontier exploration, and detailed logging."""

    # ---- Constants & thresholds ----
    RECHARGE_START = 50  # enter recharge when below this (increased for larger maps)
    RECHARGE_STOP = 90  # stay in recharge until at least this
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
        self._station_positions: Dict[str, Tuple[int, int]] = {}  # discovered stations
        self._wall_positions: set[Tuple[int, int]] = set()  # learned walls
        self._visited_cells: set[Tuple[int, int]] = set()

        # Occupancy: 0=unknown, 1=free, 2=wall
        self._occ_unknown, self._occ_free, self._occ_wall = 0, 1, 2
        self._occ = [[self._occ_unknown for _ in range(self._map_width)] for _ in range(self._map_height)]

        # Movement tracking
        self._prev_pos: Optional[Tuple[int, int]] = None
        self._last_action_idx: Optional[int] = None
        self._last_attempt_was_use: bool = False
        self._MOVE_N = self._action_lookup.get("move_north", -1)
        self._MOVE_S = self._action_lookup.get("move_south", -1)
        self._MOVE_E = self._action_lookup.get("move_east", -1)
        self._MOVE_W = self._action_lookup.get("move_west", -1)
        self._MOVE_SET = {self._MOVE_N, self._MOVE_S, self._MOVE_E, self._MOVE_W}

        # Cycle detection
        self._loop_window: List[Tuple[int, int, int]] = []  # (r,c,last_action_idx or -1)
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

    # ---------- Visual Discovery ----------
    def _discover_stations_from_observation(self, obs: MettaGridObservation, state: AgentState) -> None:
        """Discover stations and walls by observing them. Mark occupancy grid properly.

        Key rules:
        - Empty cells are walkable (free)
        - Walls are not walkable (wall)
        - Stations are not walkable (wall) - you use them from adjacent cells
        - Agents don't block movement (ignored)
        """
        if state.agent_row == -1:
            return

        type_id_feature = self._feature_name_to_id.get("type_id", 0)
        obs_height = 11
        obs_width = 11
        obs_height_radius = obs_height // 2
        obs_width_radius = obs_width // 2

        # Process all observed cells with type_id
        for tok in obs:
            if self._to_int(tok[1]) == type_id_feature:
                # Relative coords encoded in tok[0] (hi nibble=row, lo nibble=col)
                obs_r = int(self._to_int(tok[0]) >> 4)
                obs_c = int(self._to_int(tok[0]) & 0x0F)
                type_id = int(self._to_int(tok[2]))

                # Absolute
                map_r = obs_r - obs_height_radius + state.agent_row
                map_c = obs_c - obs_width_radius + state.agent_col
                if not (0 <= map_r < self._map_height and 0 <= map_c < self._map_width):
                    continue

                # Determine what this cell is
                if type_id == self._wall_type_id:
                    # It's a wall - mark as unwalkable
                    self._occ[map_r][map_c] = self._occ_wall
                    self._wall_positions.add((map_r, map_c))
                elif type_id in self._type_id_to_station:
                    # It's a station - mark as unwalkable (you can't walk onto stations)
                    station_name = self._type_id_to_station[type_id]
                    self._occ[map_r][map_c] = self._occ_wall
                    pos = (map_r, map_c)
                    if station_name not in self._station_positions:
                        self._station_positions[station_name] = pos
                        logger.info(f"Discovered {station_name} at {pos}")
                elif not self._object_type_names[type_id].startswith("agent"):
                    # It's some other object (not agent, not wall, not station) - mark as free
                    self._occ[map_r][map_c] = self._occ_free

        # Mark current agent position as free (we're standing here)
        if 0 <= state.agent_row < self._map_height and 0 <= state.agent_col < self._map_width:
            self._occ[state.agent_row][state.agent_col] = self._occ_free

    def _update_wall_knowledge(self, state: AgentState) -> None:
        """Update occupancy based on movement results.

        - If we moved successfully, mark new cell as free
        - If we tried to move but didn't, mark target as wall
        """
        if not self._prev_pos or self._last_action_idx not in self._MOVE_SET:
            return

        current_pos = (state.agent_row, state.agent_col)

        if current_pos != self._prev_pos:
            # We moved! Mark new position as free and path between as free
            self._occ[current_pos[0]][current_pos[1]] = self._occ_free
            return

        # We tried to move but didn't - target must be blocked
        dr, dc = self._action_to_dir(self._last_action_idx)
        if dr is None or dc is None:
            return

        wall_r = self._prev_pos[0] + dr
        wall_c = self._prev_pos[1] + dc
        if not (0 <= wall_r < self._map_height and 0 <= wall_c < self._map_width):
            return

        # Mark as wall (could be actual wall or station)
        if self._occ[wall_r][wall_c] != self._occ_wall:
            logger.info(f"Marking blocked cell at ({wall_r},{wall_c})")
        self._wall_positions.add((wall_r, wall_c))
        self._occ[wall_r][wall_c] = self._occ_wall

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

        # Mark current as free
        if 0 <= state.agent_row < self._map_height and 0 <= state.agent_col < self._map_width:
            self._occ[state.agent_row][state.agent_col] = self._occ_free

        # Discover stations + learn walls
        self._discover_stations_from_observation(obs, state)
        self._update_wall_knowledge(state)

        # Visited
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

        # # Log (stdout)
        # self._print_step_log(action_idx, state)

        # Stuck detection: if we keep trying the same action from the same position, force exploration
        key = (state.agent_row, state.agent_col, action_idx)
        self._loop_window.append(key)
        if len(self._loop_window) > self._loop_window_max:
            self._loop_window.pop(0)

        # If stuck in a tight loop (same state repeated), try different direction
        if self._loop_window.count(key) > 3:
            if state.step_count % 10 == 0:
                action_name = self._action_names[action_idx]
                logger.info(f"Stuck at ({state.agent_row},{state.agent_col}) action={action_name}, trying alternatives")
            # Try directions in order, avoiding the current stuck action
            for a in (self._MOVE_E, self._MOVE_W, self._MOVE_S, self._MOVE_N):
                if a in self._MOVE_SET and a != action_idx:
                    action_idx = a
                    # Clear loop window to give new direction a chance
                    self._loop_window = []
                    break

        # Bookkeeping
        self._last_action_idx = action_idx
        self._prev_pos = (state.agent_row, state.agent_col)
        state.last_heart = state.heart

        return dtype_actions.type(action_idx), state

    # ---------- Update helpers ----------
    @staticmethod
    def _to_int(x) -> int:
        try:
            return int(x)
        except Exception:
            try:
                return int(x.item())
            except Exception:
                return int(x)

    def _update_agent_position(self, state: AgentState) -> None:
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
                if self._to_int(tok[1]) == last_reward_feat:
                    state.last_reward = float(self._to_int(tok[2]))
                    state.total_reward += state.last_reward
                    break
        except Exception:
            pass

    def _has_heart_from_obs(self, obs: MettaGridObservation) -> bool:
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
        state.carbon = self._read_int_feature(obs, "inv:carbon")
        state.oxygen = self._read_int_feature(obs, "inv:oxygen")
        state.germanium = self._read_int_feature(obs, "inv:germanium")
        state.silicon = self._read_int_feature(obs, "inv:silicon")
        state.energy = self._read_int_feature(obs, "inv:energy")
        state.heart = 1 if self._has_heart_from_obs(obs) else 0

    # ---------- Occupancy / Frontier ----------
    def _neighbors4(self, r: int, c: int):
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self._map_height and 0 <= nc < self._map_width:
                yield nr, nc

    def _compute_frontiers(self) -> List[Tuple[int, int]]:
        """Unknown cells that are 4-adjacent to a known free cell."""
        fronts = []
        for r in range(self._map_height):
            for c in range(self._map_width):
                if self._occ[r][c] != self._occ_unknown:
                    continue
                for nr, nc in self._neighbors4(r, c):
                    if self._occ[nr][nc] == self._occ_free:
                        fronts.append((r, c))
                        break
        return fronts

    def _bfs_next_step_occ(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """BFS through known-free cells only; returns next cell on path or None."""
        if start == goal:
            return start
        q = deque([start])
        parent = {start: None}
        while q:
            r, c = q.popleft()
            for nr, nc in self._neighbors4(r, c):
                if (nr, nc) in parent:
                    continue
                if self._occ[nr][nc] != self._occ_free:
                    continue
                parent[(nr, nc)] = (r, c)
                if (nr, nc) == goal:
                    step = (nr, nc)
                    while parent[step] != start:
                        step = parent[step]
                    return step
                q.append((nr, nc))
        return None

    def _bfs_next_step_optimistic(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """BFS with optimistic assumption: unknown cells are walkable.

        Only avoids cells known to be walls. This allows pathfinding through
        unexplored areas, and we'll learn about walls when we bump into them.
        """
        if start == goal:
            return start
        q = deque([start])
        parent = {start: None}
        while q:
            r, c = q.popleft()
            for nr, nc in self._neighbors4(r, c):
                if (nr, nc) in parent:
                    continue
                # Skip only if definitely blocked (wall)
                if self._occ[nr][nc] == self._occ_wall:
                    continue
                parent[(nr, nc)] = (r, c)
                if (nr, nc) == goal:
                    step = (nr, nc)
                    while parent[step] != start:
                        step = parent[step]
                    return step
                q.append((nr, nc))
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
            for nr, nc in self._neighbors4(r, c):
                if (nr, nc) in fronts:
                    return (nr, nc)
            for nr, nc in self._neighbors4(r, c):
                if (nr, nc) in seen:
                    continue
                if self._occ[nr][nc] != self._occ_free:
                    continue
                seen.add((nr, nc))
                q.append((nr, nc))
        return None

    def _plan_to_frontier_action(self, state: AgentState) -> Optional[int]:
        """Plan one step toward the chosen frontier (if any)."""
        target = self._choose_frontier(state)
        if not target:
            return None

        tr, tc = target
        sr, sc = state.agent_row, state.agent_col

        # If we're already adjacent to the frontier, step toward it directly
        if abs(tr - sr) + abs(tc - sc) == 1:
            return self._step_toward(tr - sr, tc - sc)

        # Otherwise, path to a free neighbor of the frontier (excluding our current position)
        candidates = [
            (nr, nc)
            for nr, nc in self._neighbors4(tr, tc)
            if self._occ[nr][nc] == self._occ_free and (nr, nc) != (sr, sc)
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

    # ---------- Decision helpers ----------
    def _determine_phase(self, state: AgentState) -> GamePhase:
        if state.heart > 0:
            return GamePhase.DEPOSIT_HEART

        germ_needed = 5 if state.hearts_assembled == 0 else max(2, 5 - state.hearts_assembled)

        if state.current_phase == GamePhase.RECHARGE and state.energy < self.RECHARGE_STOP:
            return GamePhase.RECHARGE

        if state.energy < self.RECHARGE_START:
            return GamePhase.RECHARGE

        if (
            state.germanium >= germ_needed
            and state.carbon >= self.CARBON_REQ
            and state.oxygen >= self.OXYGEN_REQ
            and state.silicon >= self.SILICON_REQ
            and state.energy >= self.ENERGY_REQ
        ):
            return GamePhase.ASSEMBLE_HEART

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
        station = self._phase_to_station.get(state.current_phase)
        if not station:
            return self._action_lookup.get("noop", 0)

        # After deposit, step off chest
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

        # If we know the station, go there via known-free BFS
        if station in self._station_positions and state.agent_row != -1:
            tr, tc = self._station_positions[station]
            dr, dc = tr - state.agent_row, tc - state.agent_col
            manhattan_dist = abs(dr) + abs(dc)

            if manhattan_dist == 1:
                # We're adjacent to the station - use it by stepping toward it
                self._last_attempt_was_use = True
                return self._step_toward(dr, dc)

            # Path to an adjacent cell of the station (not the station itself)
            # Use optimistic assumption: unknown cells are walkable until proven otherwise
            adjacent_cells = [
                (nr, nc)
                for nr, nc in self._neighbors4(tr, tc)
                if self._occ[nr][nc] != self._occ_wall  # Not known to be blocked
            ]
            if adjacent_cells:
                # Pick the closest adjacent cell by manhattan distance
                adjacent_cells.sort(key=lambda p: abs(p[0] - state.agent_row) + abs(p[1] - state.agent_col))
                for goal in adjacent_cells:
                    step = self._bfs_next_step_optimistic((state.agent_row, state.agent_col), goal)
                    if step is not None:
                        nr, nc = step
                        self._last_attempt_was_use = False
                        return self._step_toward(nr - state.agent_row, nc - state.agent_col)

            # Not yet reachable; reveal more map
            self._last_attempt_was_use = False
            act = self._plan_to_frontier_action(state)
            if act is not None:
                return act
            return self._explore_simple(state)

        # Unknown station: explore
        self._last_attempt_was_use = False
        act = self._plan_to_frontier_action(state)
        if act is not None:
            return act
        return self._explore_simple(state)

    # ---------- Navigation primitives ----------
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

    # ---------- Exploration fallback ----------
    def _explore_simple(self, state: AgentState) -> int:
        """Boustrophedon sweep with simple loop avoidance."""
        if state.agent_row == -1:
            return self._action_lookup.get("noop", 0)

        cur = (state.agent_row, state.agent_col)
        if not self._recent_positions or self._recent_positions[-1] != cur:
            self._recent_positions.append(cur)
            if len(self._recent_positions) > self._max_recent_positions:
                self._recent_positions.pop(0)

        row_parity = state.agent_row % 2
        preferred_dir = self._MOVE_E if row_parity == 0 else self._MOVE_W

        dr, dc = self._action_to_dir(preferred_dir)
        nr, nc = state.agent_row + (dr or 0), state.agent_col + (dc or 0)
        if (
            preferred_dir in self._MOVE_SET
            and 0 <= nr < self._map_height
            and 0 <= nc < self._map_width
            and (nr, nc) not in self._wall_positions
        ):
            return preferred_dir

        down_r, down_c = state.agent_row + 1, state.agent_col
        if 0 <= down_r < self._map_height and (down_r, down_c) not in self._wall_positions:
            return self._MOVE_S if self._MOVE_S != -1 else self._action_lookup.get("noop", 0)

        alt = self._find_best_exploration_direction(state)
        if alt is not None:
            return alt

        return preferred_dir if preferred_dir != -1 else self._action_lookup.get("noop", 0)

    def _find_best_exploration_direction(self, state: AgentState) -> Optional[int]:
        directions = [
            (self._MOVE_N, (-1, 0)),
            (self._MOVE_S, (1, 0)),
            (self._MOVE_E, (0, 1)),
            (self._MOVE_W, (0, -1)),
        ]
        best_action, best_score = None, -1
        recent = self._recent_positions

        for action, (dr, dc) in directions:
            nr, nc = state.agent_row + dr, state.agent_col + dc
            if not (0 <= nr < self._map_height and 0 <= nc < self._map_width):
                continue
            np = (nr, nc)
            if np in self._wall_positions:
                continue
            score = 10 if np not in recent else recent.index(np)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    # ---------- Logging ----------
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
