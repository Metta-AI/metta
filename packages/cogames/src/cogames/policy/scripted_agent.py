"""Scripted agent policy for CoGames training facility missions with detailed logging."""

from __future__ import annotations

import logging
import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from cogames.policy.interfaces import AgentPolicy, Policy, StatefulAgentPolicy, StatefulPolicyImpl
from mettagrid import MettaGridAction, MettaGridEnv, MettaGridObservation, dtype_actions
from mettagrid.mettagrid_c import PackedCoordinate

logger = logging.getLogger("cogames.policy.scripted_agent")


class GamePhase(Enum):
    """Game phases for strategic play."""

    GATHER_GERMANIUM = "gather_germanium"
    GATHER_SILICON = "gather_silicon"
    GATHER_CARBON = "gather_carbon"
    GATHER_OXYGEN = "gather_oxygen"
    ASSEMBLE_HEART = "assemble_heart"
    DEPOSIT_HEART = "deposit_heart"
    RECHARGE = "recharge"


@dataclass
class AgentState:
    """State for a scripted agent."""

    current_phase: GamePhase = GamePhase.GATHER_GERMANIUM
    current_glyph: str = "default"

    # Inventory tracking
    carbon: int = 0
    oxygen: int = 0
    germanium: int = 0
    silicon: int = 0
    energy: int = 100
    heart: int = 0

    # Strategy tracking
    hearts_assembled: int = 0
    wait_counter: int = 0

    # Position tracking (absolute grid coordinates)
    agent_row: int = -1
    agent_col: int = -1

    step_count: int = 0


class ScriptedAgentPolicyImpl(StatefulPolicyImpl[AgentState]):
    """Scripted policy with absolute position tracking and detailed logging."""

    def __init__(self, env: MettaGridEnv):
        """Initialize the scripted policy."""
        self._env = env
        self._action_names = env.action_names
        self._object_type_names = env.object_type_names

        # Build action lookup
        self._action_lookup = {name: idx for idx, name in enumerate(self._action_names)}

        # Build feature ID lookup
        obs_features = env.observation_features
        self._feature_name_to_id: dict[str, int] = {feature.name: feature.id for feature in obs_features.values()}

        # Build object type mapping
        self._object_name_to_type_id: dict[str, int] = {name: idx for idx, name in enumerate(self._object_type_names)}

        # Build glyph mapping
        from cogames.cogs_vs_clips.vibes import VIBES

        self._glyph_name_to_id: dict[str, int] = {vibe.name: idx for idx, vibe in enumerate(VIBES)}

        # Station to glyph mapping
        self._station_to_glyph: dict[str, str] = {
            "charger": "charger",
            "carbon_extractor": "carbon",
            "oxygen_extractor": "oxygen",
            "germanium_extractor": "germanium",
            "silicon_extractor": "silicon",
            # For assembling a heart, glyph must be 'heart'
            "assembler": "heart",
            "chest": "chest",
        }

        # Get station absolute positions from environment
        self._station_positions = self._get_station_positions()

        # Get wall positions and map dimensions
        self._wall_positions = self._get_wall_positions()
        self._map_height = env.c_env.map_height
        self._map_width = env.c_env.map_width

        # Trajectory logging setup
        logs_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._traj_log_path = os.path.join(logs_dir, f"scripted_agent_detailed_{timestamp}.csv")
        try:
            with open(self._traj_log_path, "w", encoding="utf-8") as f:
                f.write(
                    "step,phase,glyph,action,agent_row,agent_col,target_station,target_row,target_col,rel_to_target,energy,C,O,G,S,H,notes\n"
                )
            logger.info(f"Detailed trajectory logging to {self._traj_log_path}")
        except Exception as e:
            logger.warning(f"Failed to init trajectory log: {e}")

        logger.info("Scripted agent initialized with position tracking")
        logger.info(f"Map size: {self._map_height}x{self._map_width}")
        logger.info(f"Station positions: {self._station_positions}")
        logger.info(f"Found {len(self._wall_positions)} walls")

    def _get_station_positions(self) -> dict[str, tuple[int, int]]:
        """Get absolute positions of all stations from the environment."""
        positions = {}
        try:
            # Query grid objects to get absolute positions
            grid_objs = self._env.c_env.grid_objects()
            for _obj_id, obj_data in grid_objs.items():
                # Skip agents - they have agent_id key
                if "agent_id" in obj_data:
                    continue
                type_id = obj_data.get("type_id")
                row = obj_data.get("r")  # Use "r" not "row"
                col = obj_data.get("c")  # Use "c" not "col"
                if type_id is not None and row is not None and col is not None:
                    type_name = self._object_type_names[type_id] if type_id < len(self._object_type_names) else None
                    if type_name in self._station_to_glyph:
                        positions[type_name] = (row, col)
        except Exception as e:
            logger.warning(f"Could not get station positions: {e}")
        return positions

    def _get_wall_positions(self) -> set[tuple[int, int]]:
        """Get absolute positions of all walls from the environment."""
        walls = set()
        try:
            grid_objs = self._env.c_env.grid_objects()
            for _obj_id, obj_data in grid_objs.items():
                type_id = obj_data.get("type_id")
                if type_id is not None:
                    type_name = self._object_type_names[type_id] if type_id < len(self._object_type_names) else None
                    if type_name == "wall":
                        row = obj_data.get("r")
                        col = obj_data.get("c")
                        if row is not None and col is not None:
                            walls.add((row, col))
        except Exception as e:
            logger.warning(f"Could not get wall positions: {e}")
        return walls

    def agent_state(self) -> AgentState:
        """Get the initial state for a new agent."""
        return AgentState()

    def step_with_state(
        self, obs: MettaGridObservation, state: Optional[AgentState]
    ) -> tuple[MettaGridAction, Optional[AgentState]]:
        """React to current observation with detailed logging."""
        if state is None:
            state = self.agent_state()

        state.step_count += 1

        # Update inventory and position from observation
        self._update_inventory(obs, state)
        self._update_agent_position(obs, state)

        # Find objects in current view
        visible_objects = self._parse_visible_objects(obs)

        # Check if we collected resources and should move to next phase
        if state.wait_counter > 0:
            # We were waiting/gathering - check if we got what we needed
            got_resources = self._check_inventory_progress(state)
            print(
                f"Step {state.step_count}: wait_counter={state.wait_counter}, got_resources={got_resources}, G={state.germanium}, phase={state.current_phase.name}"
            )
            if got_resources:
                # Move to next phase only after resource progress
                print(
                    f"Step {state.step_count}: Moving to next phase! (got={got_resources}, waited={state.wait_counter})"
                )
                logger.info(f"Step {state.step_count}: Moving to next phase after wait={state.wait_counter}")
                state.wait_counter = 0
                if state.current_phase == GamePhase.ASSEMBLE_HEART:
                    state.hearts_assembled += 1

        # Determine phase
        state.current_phase = self._determine_phase(state)

        # Execute phase
        action_idx = self._execute_phase(visible_objects, state)

        # Log detailed trajectory
        self._log_trajectory(action_idx, visible_objects, state)

        return dtype_actions.type(action_idx), state

    def _update_agent_position(self, obs: MettaGridObservation, state: AgentState) -> None:
        """Update agent's absolute position from observation."""
        # Try to get absolute position from grid objects
        try:
            grid_objs = self._env.c_env.grid_objects()
            for _obj_id, obj_data in grid_objs.items():
                # Look for agent by checking agent_id key (agents have this, stations don't)
                if "agent_id" in obj_data and obj_data.get("agent_id") == 0:  # First agent
                    state.agent_row = obj_data.get("r", -1)  # Use "r" not "row"
                    state.agent_col = obj_data.get("c", -1)  # Use "c" not "col"
                    break
        except Exception as e:
            logger.debug(f"Could not update agent position: {e}")

    def _update_inventory(self, obs: MettaGridObservation, state: AgentState) -> None:
        """Update inventory from observation."""
        inv_map = {
            "inv:carbon": "carbon",
            "inv:oxygen": "oxygen",
            "inv:germanium": "germanium",
            "inv:silicon": "silicon",
            "inv:energy": "energy",
            "inv:heart": "heart",
        }

        old_inv = {k: getattr(state, k) for k in ["carbon", "oxygen", "germanium", "silicon", "energy", "heart"]}

        for feat_name, attr_name in inv_map.items():
            feat_id = self._feature_name_to_id.get(feat_name)
            if feat_id is not None:
                for token in obs:
                    if token[0] == 255:
                        break
                    if token[1] == feat_id:
                        setattr(state, attr_name, int(token[2]))
                        break

        # Log inventory changes
        if state.step_count <= 50:
            for resource in ["carbon", "oxygen", "germanium", "silicon", "heart", "energy"]:
                new_val = getattr(state, resource)
                if new_val != old_inv[resource]:
                    logger.info(f"Step {state.step_count}: 📦 {resource} {old_inv[resource]} -> {new_val}")

    def _parse_visible_objects(self, obs: MettaGridObservation) -> dict[str, list[tuple[int, int]]]:
        """Find all visible objects by type. Returns positions relative to agent (row, col)."""
        type_id_feat = self._feature_name_to_id.get("type_id")
        if type_id_feat is None:
            return {}

        objects_by_name: dict[str, list[tuple[int, int]]] = {}

        for token in obs:
            if token[0] == 255:
                break

            if token[1] == type_id_feat:
                coords = PackedCoordinate.unpack(token[0])
                if coords:
                    type_id = int(token[2])
                    if type_id < len(self._object_type_names):
                        type_name = self._object_type_names[type_id]
                        if type_name not in objects_by_name:
                            objects_by_name[type_name] = []
                        objects_by_name[type_name].append(coords)

        return objects_by_name

    def _determine_phase(self, state: AgentState) -> GamePhase:
        """Determine next phase based on inventory."""
        if state.heart > 0:
            return GamePhase.DEPOSIT_HEART

        germ_needed = 5 if state.hearts_assembled == 0 else max(2, 5 - state.hearts_assembled)

        if (
            state.germanium >= germ_needed
            and state.carbon >= 20
            and state.oxygen >= 20
            and state.silicon >= 50
            and state.energy >= 20
        ):
            return GamePhase.ASSEMBLE_HEART

        if state.energy < 40:
            return GamePhase.RECHARGE

        if state.germanium < germ_needed:
            return GamePhase.GATHER_GERMANIUM
        elif state.silicon < 50:
            return GamePhase.GATHER_SILICON
        elif state.carbon < 20:
            return GamePhase.GATHER_CARBON
        elif state.oxygen < 20:
            return GamePhase.GATHER_OXYGEN
        else:
            return GamePhase.GATHER_GERMANIUM  # Cycle back

    def _execute_phase(self, visible_objects: dict[str, list[tuple[int, int]]], state: AgentState) -> int:
        """Execute current phase."""
        phase_to_station = {
            GamePhase.GATHER_GERMANIUM: "germanium_extractor",
            GamePhase.GATHER_SILICON: "silicon_extractor",
            GamePhase.GATHER_CARBON: "carbon_extractor",
            GamePhase.GATHER_OXYGEN: "oxygen_extractor",
            GamePhase.ASSEMBLE_HEART: "assembler",
            GamePhase.DEPOSIT_HEART: "chest",
            GamePhase.RECHARGE: "charger",
        }

        target_station = phase_to_station.get(state.current_phase)
        if not target_station:
            return self._action_lookup.get("noop", 0)

        target_glyph = self._station_to_glyph.get(target_station, "default")

        # Step 1: Change glyph if needed
        if state.current_glyph != target_glyph:
            state.current_glyph = target_glyph
            state.wait_counter = 0
            glyph_id = self._glyph_name_to_id.get(target_glyph, 0)
            action_name = f"change_glyph_{glyph_id}"
            if state.step_count <= 20:
                logger.info(f"Step {state.step_count}: Changing glyph to {target_glyph}")
            return self._action_lookup.get(action_name, self._action_lookup.get("noop", 0))

        # Step 2: Interact by ADJACENCY: when within manhattan distance 1, wait/noop to trigger interaction
        if target_station in self._station_positions and state.agent_row != -1:
            target_row, target_col = self._station_positions[target_station]
            dr = target_row - state.agent_row
            dc = target_col - state.agent_col
            manhattan = abs(dr) + abs(dc)
            if manhattan == 1:
                # When adjacent, attempt to MOVE INTO the station tile to trigger interaction
                if dr > 0:
                    return self._action_lookup.get("move_south", 0)
                if dr < 0:
                    return self._action_lookup.get("move_north", 0)
                if dc > 0:
                    return self._action_lookup.get("move_east", 0)
                if dc < 0:
                    return self._action_lookup.get("move_west", 0)
            elif manhattan == 0:
                # Already on tile; noop to allow interaction if applicable
                state.wait_counter += 1
                return self._action_lookup.get("noop", 0)

        # Step 3: Default navigation using BFS
        return self._navigate_to_station(target_station, state)

    def _move_toward(self, target: tuple[int, int]) -> int:
        """Move one step toward target (relative coords)."""
        row, col = target

        # Move row first if non-zero
        if row > 0:
            return self._action_lookup.get("move_south", 0)
        elif row < 0:
            return self._action_lookup.get("move_north", 0)

        # Then move col
        if col > 0:
            return self._action_lookup.get("move_east", 0)
        elif col < 0:
            return self._action_lookup.get("move_west", 0)

        return self._action_lookup.get("noop", 0)

    def _navigate_to_station(self, station_name: str, state: AgentState) -> int:
        """Navigate to station using BFS pathfinding to avoid walls."""
        if station_name not in self._station_positions:
            return self._action_lookup.get("noop", 0)

        target_row, target_col = self._station_positions[station_name]

        if state.agent_row == -1 or state.agent_col == -1:
            # Don't know our position, use heuristic
            return self._fallback_navigate(state.current_phase)

        # Use BFS to find path avoiding walls
        path = self._bfs_pathfind((state.agent_row, state.agent_col), (target_row, target_col))

        if path and len(path) > 1:
            # Get next step in path
            next_row, next_col = path[1]
            row_diff = next_row - state.agent_row
            col_diff = next_col - state.agent_col

            # Convert to action
            if row_diff > 0:
                return self._action_lookup.get("move_south", 0)
            elif row_diff < 0:
                return self._action_lookup.get("move_north", 0)
            elif col_diff > 0:
                return self._action_lookup.get("move_east", 0)
            elif col_diff < 0:
                return self._action_lookup.get("move_west", 0)

        return self._action_lookup.get("noop", 0)

    def _bfs_pathfind(self, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
        """BFS pathfinding that avoids walls and non-goal stations. Returns list of (row, col) positions."""
        if start == goal:
            return [start]

        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            (row, col), path = queue.popleft()

            # Try all 4 directions
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                next_row, next_col = row + dr, col + dc

                # Check bounds
                if not (0 <= next_row < self._map_height and 0 <= next_col < self._map_width):
                    continue

                # Block walls
                if (next_row, next_col) in self._wall_positions:
                    continue

                # Block other stations to avoid trying to walk through them
                if (next_row, next_col) in self._station_positions.values() and (next_row, next_col) != goal:
                    continue

                # Check if visited
                if (next_row, next_col) in visited:
                    continue

                visited.add((next_row, next_col))
                new_path = path + [(next_row, next_col)]

                # Check if goal
                if (next_row, next_col) == goal:
                    return new_path

                queue.append(((next_row, next_col), new_path))

        # No path found
        return []

    # Note: helper for adjacency paths removed since stations require stepping onto tiles

    def _fallback_navigate(self, phase: GamePhase) -> int:
        """Fallback navigation when position unknown."""
        # Simple heuristic: germanium/silicon are bottom, carbon/oxygen are top
        if phase in (GamePhase.GATHER_GERMANIUM, GamePhase.GATHER_SILICON, GamePhase.DEPOSIT_HEART):
            return self._action_lookup.get("move_south", 0)
        elif phase in (GamePhase.GATHER_CARBON, GamePhase.GATHER_OXYGEN):
            return self._action_lookup.get("move_north", 0)
        else:
            return self._action_lookup.get("move_south", 0)

    def _check_inventory_progress(self, state: AgentState) -> bool:
        """Check if we made progress on current phase."""
        phase = state.current_phase
        if phase == GamePhase.GATHER_GERMANIUM:
            return state.germanium > 0
        elif phase == GamePhase.GATHER_SILICON:
            return state.silicon > 0
        elif phase == GamePhase.GATHER_CARBON:
            return state.carbon > 0
        elif phase == GamePhase.GATHER_OXYGEN:
            return state.oxygen > 0
        elif phase == GamePhase.ASSEMBLE_HEART:
            return state.heart > 0
        elif phase == GamePhase.DEPOSIT_HEART:
            return state.heart == 0
        return True

    def _log_trajectory(self, action_idx: int, visible_objects: dict, state: AgentState) -> None:
        """Write detailed trajectory log."""
        try:
            action_name = self._action_names[action_idx] if action_idx < len(self._action_names) else "?"

            phase_to_station = {
                GamePhase.GATHER_GERMANIUM: "germanium_extractor",
                GamePhase.GATHER_SILICON: "silicon_extractor",
                GamePhase.GATHER_CARBON: "carbon_extractor",
                GamePhase.GATHER_OXYGEN: "oxygen_extractor",
                GamePhase.ASSEMBLE_HEART: "assembler",
                GamePhase.DEPOSIT_HEART: "chest",
                GamePhase.RECHARGE: "charger",
            }
            target_station = phase_to_station.get(state.current_phase, "unknown")

            target_row, target_col = self._station_positions.get(target_station, (-1, -1))

            rel_to_target = "?"
            if state.agent_row != -1 and target_row != -1:
                rel_row = target_row - state.agent_row
                rel_col = target_col - state.agent_col
                rel_to_target = f"({rel_row},{rel_col})"

            notes = f"wait={state.wait_counter}"

            with open(self._traj_log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{state.step_count},{state.current_phase.value},{state.current_glyph},{action_name},"
                    f"{state.agent_row},{state.agent_col},{target_station},{target_row},{target_col},{rel_to_target},"
                    f"{state.energy},{state.carbon},{state.oxygen},{state.germanium},{state.silicon},{state.heart},{notes}\n"
                )
        except Exception as e:
            logger.debug(f"Failed to write trajectory log: {e}")


class ScriptedAgentPolicy(Policy):
    """Scripted policy for training facility missions."""

    def __init__(self, env: MettaGridEnv, device=None):
        """Initialize the scripted policy."""
        self._env = env
        self._impl = ScriptedAgentPolicyImpl(env)

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        """Get an AgentPolicy instance for a specific agent."""
        return StatefulAgentPolicy(self._impl, agent_id)
