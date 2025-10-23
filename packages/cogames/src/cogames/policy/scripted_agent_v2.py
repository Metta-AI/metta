"""Optimized scripted agent with visual discovery and smart navigation.

Key features:
- Visual station discovery from observations (type_id feature)
- Incremental wall discovery during movement
- BFS pathfinding with known walls once stations are discovered
- Loop avoidance during exploration
- No omniscience - discovers the map through interaction
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from cogames.policy.interfaces import AgentPolicy, Policy, StatefulAgentPolicy, StatefulPolicyImpl
from mettagrid import MettaGridAction, MettaGridEnv, MettaGridObservation, dtype_actions

logger = logging.getLogger("cogames.policy.scripted_agent")


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

    # Position
    agent_row: int = -1
    agent_col: int = -1

    # Bookkeeping
    step_count: int = 0
    last_heart: int = 0
    last_reward: float = 0.0
    total_reward: float = 0.0
    hearts_assembled: int = 0
    just_deposited: bool = False


class ScriptedAgentPolicyImpl(StatefulPolicyImpl[AgentState]):
    RECHARGE_THRESHOLD = 40
    CARBON_REQ = 20
    OXYGEN_REQ = 20
    GERMANIUM_REQ = 1
    SILICON_REQ = 1
    HEART_FEATURE_NAME = "inv:heart"
    HEART_SENTINEL_FIRST_FIELD = 85  # 0x55

    _station_to_glyph: Dict[str, str] = {
        "charger": "charger",
        "carbon_extractor": "carbon",
        "oxygen_extractor": "oxygen",
        "germanium_extractor": "germanium",
        "silicon_extractor": "silicon",
        "assembler": "heart",
        "chest": "chest",
    }

    _phase_to_station: Dict[GamePhase, str] = {
        GamePhase.GATHER_GERMANIUM: "germanium_extractor",
        GamePhase.GATHER_SILICON: "silicon_extractor",
        GamePhase.GATHER_CARBON: "carbon_extractor",
        GamePhase.GATHER_OXYGEN: "oxygen_extractor",
        GamePhase.ASSEMBLE_HEART: "assembler",
        GamePhase.DEPOSIT_HEART: "chest",
        GamePhase.RECHARGE: "charger",
    }

    def __init__(self, env: MettaGridEnv):
        self._env = env
        self._action_names: List[str] = env.action_names
        self._action_lookup: Dict[str, int] = {name: i for i, name in enumerate(self._action_names)}
        self._feature_name_to_id: Dict[str, int] = {
            feature.name: feature.id for feature in env.observation_features.values()
        }

        # Build type_id -> station name mapping for visual discovery
        self._type_id_to_station: Dict[int, str] = {}
        for type_id, name in enumerate(env.object_type_names):
            if name and name != "wall" and not name.startswith("agent"):
                self._type_id_to_station[type_id] = name

        # Glyphs
        from cogames.cogs_vs_clips.vibes import VIBES
        self._glyph_name_to_id: Dict[str, int] = {vibe.name: idx for idx, vibe in enumerate(VIBES)}

        # Movement actions
        self._MOVE_N = self._action_lookup.get("move_north", -1)
        self._MOVE_S = self._action_lookup.get("move_south", -1)
        self._MOVE_E = self._action_lookup.get("move_east", -1)
        self._MOVE_W = self._action_lookup.get("move_west", -1)
        self._MOVE_SET = {self._MOVE_N, self._MOVE_S, self._MOVE_E, self._MOVE_W}

        # Map bounds
        self._H = env.c_env.map_height
        self._W = env.c_env.map_width

        # Incremental knowledge
        self._discovered_stations: Dict[str, Tuple[int, int]] = {}
        self._known_walls: Set[Tuple[int, int]] = set()

        # Movement tracking
        self._last_action_idx: Optional[int] = None
        self._prev_pos: Optional[Tuple[int, int]] = None

        # Exploration state
        self._exploration_direction: int = self._MOVE_E
        self._stuck_counter: int = 0
        self._recent_positions: List[Tuple[int, int]] = []
        self._max_recent_positions: int = 10

        logger.info("Scripted agent (visual discovery + smart navigation) initialized")
        logger.info(f"Map size: {self._H}x{self._W}")

    def agent_state(self) -> AgentState:
        return AgentState()

    def step_with_state(self, obs: MettaGridObservation, state: Optional[AgentState]) -> tuple[MettaGridAction, Optional[AgentState]]:
        if state is None:
            state = self.agent_state()

        state.step_count += 1

        # Update from observation
        self._update_inventory(obs, state)
        self._update_agent_position(state)
        self._update_rewards(obs, state)
        self._discover_stations_from_observation(obs, state)

        # Mark walls from previous move attempt
        self._update_wall_knowledge(state)

        # Deposit detection
        if (state.last_heart > 0 and state.heart == 0) or (
            state.current_phase == GamePhase.DEPOSIT_HEART and state.last_reward > 0
        ):
            state.just_deposited = True

        # Decide phase & act
        state.current_phase = self._determine_phase(state)
        action_idx = self._execute_phase(state)

        # Bookkeeping
        self._prev_pos = (state.agent_row, state.agent_col)
        self._last_action_idx = action_idx
        state.last_heart = state.heart

        return dtype_actions.type(action_idx), state

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

        if not self._in_bounds(wall_r, wall_c):
            return

        # Don't mark stations as walls
        wall_pos = (wall_r, wall_c)
        is_station = wall_pos in self._discovered_stations.values()
        if not is_station:
            self._known_walls.add(wall_pos)

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

                    if station_name not in self._discovered_stations:
                        self._discovered_stations[station_name] = pos
                        logger.info(f"Discovered {station_name} at {pos}")

    def _update_agent_position(self, state: AgentState) -> None:
        """Get agent's absolute position from environment."""
        try:
            for _obj_id, obj in self._env.c_env.grid_objects().items():
                if obj.get("agent_id") == 0:
                    state.agent_row = obj.get("r", -1)
                    state.agent_col = obj.get("c", -1)
                    break
        except Exception:
            pass

    def _update_rewards(self, obs: MettaGridObservation, state: AgentState) -> None:
        try:
            last_reward_feat = self._feature_name_to_id.get("global:last_reward")
            if last_reward_feat is None:
                return
            for tok in obs:
                if int(tok[1]) == last_reward_feat:
                    state.last_reward = float(int(tok[2]))
                    state.total_reward += state.last_reward
                    break
        except Exception:
            pass

    def _has_heart_from_obs(self, obs: MettaGridObservation) -> bool:
        """Heart present only if FIRST 'inv:heart' token's first field == 85."""
        heart_fid = self._feature_name_to_id.get(self.HEART_FEATURE_NAME)
        if heart_fid is None:
            return False

        for tok in obs:
            if int(tok[1]) == heart_fid:
                return int(tok[0]) == self.HEART_SENTINEL_FIRST_FIELD

        return False

    def _read_int_feature(self, obs: MettaGridObservation, feat_name: str) -> int:
        fid = self._feature_name_to_id.get(feat_name)
        if fid is None:
            return 0
        for tok in obs:
            if int(tok[1]) == fid:
                return int(tok[2])
        return 0

    def _update_inventory(self, obs: MettaGridObservation, state: AgentState) -> None:
        state.carbon = self._read_int_feature(obs, "inv:carbon")
        state.oxygen = self._read_int_feature(obs, "inv:oxygen")
        state.germanium = self._read_int_feature(obs, "inv:germanium")
        state.silicon = self._read_int_feature(obs, "inv:silicon")
        state.energy = self._read_int_feature(obs, "inv:energy")
        state.heart = 1 if self._has_heart_from_obs(obs) else 0

    def _determine_phase(self, state: AgentState) -> GamePhase:
        if state.heart > 0:
            return GamePhase.DEPOSIT_HEART

        germ_needed = self.GERMANIUM_REQ
        sil_needed = self.SILICON_REQ

        if state.energy < self.RECHARGE_THRESHOLD:
            return GamePhase.RECHARGE

        if state.germanium < germ_needed:
            return GamePhase.GATHER_GERMANIUM
        if state.silicon < sil_needed:
            return GamePhase.GATHER_SILICON

        if state.carbon >= self.CARBON_REQ and state.oxygen >= self.OXYGEN_REQ:
            return GamePhase.ASSEMBLE_HEART

        if state.carbon < self.CARBON_REQ:
            return GamePhase.GATHER_CARBON
        if state.oxygen < self.OXYGEN_REQ:
            return GamePhase.GATHER_OXYGEN

        return GamePhase.GATHER_GERMANIUM

    def _execute_phase(self, state: AgentState) -> int:
        station = self._phase_to_station[state.current_phase]

        # Move away after deposit
        if state.just_deposited:
            state.just_deposited = False

        # Glyph switching
        needed_glyph = self._station_to_glyph.get(station, "default")
        if state.current_glyph != needed_glyph:
            state.current_glyph = needed_glyph
            glyph_id = self._glyph_name_to_id.get(needed_glyph, 0)
            return self._action_lookup.get(f"change_glyph_{glyph_id}", self._noop())

        # Navigate to station
        if station in self._discovered_stations and state.agent_row != -1:
            tr, tc = self._discovered_stations[station]
            dr, dc = tr - state.agent_row, tc - state.agent_col

            # Adjacent: step into station
            if abs(dr) + abs(dc) == 1:
                return self._step_toward(dr, dc)

            # Use BFS if we have wall knowledge, otherwise greedy
            if len(self._known_walls) > 5:  # Have enough wall knowledge
                next_step = self._bfs_next_step((state.agent_row, state.agent_col), (tr, tc))
                if next_step is not None:
                    return self._step_toward(next_step[0] - state.agent_row, next_step[1] - state.agent_col)

            # Greedy navigation with exploration fallback
            if self._prev_pos and (state.agent_row, state.agent_col) == self._prev_pos and self._last_action_idx in self._MOVE_SET:
                # Got stuck, use exploration to navigate around
                return self._explore_simple(state)

            # Greedy: move on primary axis
            if abs(dr) > abs(dc):
                return self._MOVE_S if dr > 0 else self._MOVE_N
            else:
                return self._MOVE_E if dc > 0 else self._MOVE_W

        # Station not discovered yet - explore
        return self._explore_simple(state)

    def _explore_simple(self, state: AgentState) -> int:
        """Simple exploration with loop avoidance."""
        if state.agent_row == -1:
            return self._noop()

        # Track position
        current_pos = (state.agent_row, state.agent_col)
        if not self._recent_positions or self._recent_positions[-1] != current_pos:
            self._recent_positions.append(current_pos)
            if len(self._recent_positions) > self._max_recent_positions:
                self._recent_positions.pop(0)

        # If stuck, try different directions
        if self._prev_pos and self._last_action_idx in self._MOVE_SET:
            if current_pos == self._prev_pos:
                self._stuck_counter += 1

                # If stuck multiple times, find direction away from recent positions
                if self._stuck_counter > 2:
                    best_action = self._find_best_exploration_direction(state)
                    if best_action is not None:
                        self._stuck_counter = 0
                        return best_action

                # Cycle through directions
                if self._exploration_direction == self._MOVE_E:
                    self._exploration_direction = self._MOVE_S
                elif self._exploration_direction == self._MOVE_S:
                    self._exploration_direction = self._MOVE_W
                elif self._exploration_direction == self._MOVE_W:
                    self._exploration_direction = self._MOVE_N
                elif self._exploration_direction == self._MOVE_N:
                    self._exploration_direction = self._MOVE_E
            else:
                self._stuck_counter = 0

        return self._exploration_direction

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

            if not self._in_bounds(nr, nc):
                continue

            next_pos = (nr, nc)
            # Avoid known walls
            if next_pos in self._known_walls:
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

    def _bfs_next_step(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """BFS pathfinding avoiding known walls."""
        if start == goal:
            return start

        q = deque([start])
        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}

        while q:
            r, c = q.popleft()
            for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                nr, nc = r + dr, c + dc
                nxt = (nr, nc)

                if not self._in_bounds(nr, nc):
                    continue
                if nxt in self._known_walls:
                    continue
                if nxt in parent:
                    continue

                parent[nxt] = (r, c)

                if nxt == goal:
                    # Reconstruct first step
                    step = nxt
                    while parent[step] != start:
                        step = parent[step]
                    return step

                q.append(nxt)

        return None

    def _noop(self) -> int:
        return self._action_lookup.get("noop", 0)

    def _step_toward(self, dr: int, dc: int) -> int:
        if dr > 0:
            return self._MOVE_S
        if dr < 0:
            return self._MOVE_N
        if dc > 0:
            return self._MOVE_E
        if dc < 0:
            return self._MOVE_W
        return self._noop()

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

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self._H and 0 <= c < self._W


class ScriptedAgentPolicy(Policy):
    """Optimized scripted policy."""

    def __init__(self, env: MettaGridEnv, device=None):
        self._env = env
        self._impl = ScriptedAgentPolicyImpl(env)

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        return StatefulAgentPolicy(self._impl, agent_id)

