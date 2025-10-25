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
    RECHARGE_STOP = 90   # stay in recharge until at least this
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
        self._station_positions: Dict[str, Tuple[int, int]] = {}  # discovered stations
        self._wall_positions: set[Tuple[int, int]] = set()        # learned walls
        self._visited_cells: set[Tuple[int, int]] = set()

        # Occupancy grid: 0=unknown, 1=free, 2=wall
        self._occ = [[self.OCC_UNKNOWN for _ in range(self._map_width)] for _ in range(self._map_height)]

        # Movement tracking
        self._prev_pos: Optional[Tuple[int, int]] = None
        self._last_action_idx: Optional[int] = None
        self._last_attempt_was_use: bool = False

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
            + str([(k, self._feature_name_to_id.get(k))
                   for k in ("inv:carbon", "inv:oxygen", "inv:germanium", "inv:silicon", "inv:energy", "inv:heart")])
        )

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

        # Mark current as free and update map from observation
        self._mark_cell(state.agent_row, state.agent_col, self.OCC_FREE)
        self._discover_stations_from_observation(obs, state)
        self._update_wall_knowledge(state)

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

        # Assembling possible?
        if (
            state.germanium >= germ_needed
            and state.carbon >= self.CARBON_REQ
            and state.oxygen >= self.OXYGEN_REQ
            and state.silicon >= self.SILICON_REQ
            and state.energy >= self.ENERGY_REQ
        ):
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

        # If station known: try to use it or move toward an adjacent cell
        if station in self._station_positions and state.agent_row != -1:
            tr, tc = self._station_positions[station]
            dr, dc = tr - state.agent_row, tc - state.agent_col
            mdist = abs(dr) + abs(dc)

            # Adjacent: attempt to step into the station (use)
            if mdist == 1:
                self._last_attempt_was_use = True
                return self._step_toward(dr, dc)

            # Otherwise: path to a non-blocked neighbor of the station (optimistic BFS)
            adj = [(nr, nc) for nr, nc in self._neighbors4(tr, tc) if self._occ[nr][nc] != self.OCC_WALL]
            if adj:
                adj.sort(key=lambda p: abs(p[0] - state.agent_row) + abs(p[1] - state.agent_col))
                for goal in adj:
                    step = self._bfs_next_step_optimistic((state.agent_row, state.agent_col), goal)
                    if step is not None:
                        nr, nc = step
                        self._last_attempt_was_use = False
                        return self._step_toward(nr - state.agent_row, nc - state.agent_col)

            # Not reachable yet: reveal more map
            self._last_attempt_was_use = False
            plan = self._plan_to_frontier_action(state)
            return plan if plan is not None else self._explore_simple(state)

        # Unknown station: explore (frontier-first, then sweep)
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
        candidates = [(nr, nc) for nr, nc in self._neighbors4(tr, tc) if self._occ[nr][nc] == self.OCC_FREE and (nr, nc) != (sr, sc)]
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

    def _bfs_next_step(self, start: Tuple[int, int], goal: Tuple[int, int], optimistic: bool) -> Optional[Tuple[int, int]]:
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
        if (
            preferred_dir in self._MOVE_SET
            and self._is_valid_position(nr, nc)
            and (nr, nc) not in self._wall_positions
        ):
            return preferred_dir

        # Try moving down a row
        down_r, down_c = state.agent_row + 1, state.agent_col
        if self._is_valid_position(down_r, down_c) and (down_r, down_c) not in self._wall_positions:
            return self._MOVE_S if self._MOVE_S != -1 else self._action_lookup.get("noop", 0)

        # Otherwise pick any reasonable alternative, preferring not-recent cells
        alt = self._find_best_exploration_direction(state)
        return alt if alt is not None else (preferred_dir if preferred_dir != -1 else self._action_lookup.get("noop", 0))

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
