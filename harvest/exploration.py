"""Exploration management for harvest policy.

Handles corridor detection, dead-end avoidance, and breadth-first exploration.
"""
from __future__ import annotations

import random
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .map import MapManager, MapCellType
    from .harvest_policy import HarvestState


class ExplorationManager:
    """Manages exploration strategy with corridor detection and dead-end avoidance."""

    def __init__(self, obs_hr: int, obs_wr: int, tag_names: dict):
        """Initialize exploration manager.

        Args:
            obs_hr: Observation half-height radius
            obs_wr: Observation half-width radius
            tag_names: Mapping of tag IDs to names
        """
        self._obs_hr = obs_hr
        self._obs_wr = obs_wr
        self._tag_names = tag_names

    def choose_exploration_direction(
        self,
        state: HarvestState,
        nearest_charger: Optional[tuple[int, int]] = None,
        max_safe_distance: int = 20
    ) -> str:
        """Choose best direction for exploration.

        Uses corridor detection and dead-end avoidance to make smart choices.
        Prefers breadth-first exploration (expanding circles from charger).

        Args:
            state: Current agent state
            nearest_charger: Position of nearest charger (if any)
            max_safe_distance: Maximum distance from charger to explore

        Returns:
            Direction to move ("north", "south", "east", "west")
        """
        dir_offsets = {"north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1)}

        # Step 1: Find passable directions (not blocked, not dead-ends)
        passable_directions = []
        for direction in ["north", "south", "east", "west"]:
            if not self._is_direction_clear(state, direction):
                continue  # Blocked by wall or obstacle

            dr, dc = dir_offsets[direction]
            target_r, target_c = state.row + dr, state.col + dc

            # Skip dead-ends
            if (target_r, target_c) in state.dead_end_positions:
                continue

            passable_directions.append((direction, target_r, target_c))

        if not passable_directions:
            # All directions blocked or lead to dead-ends - noop
            return "north"  # Default fallback

        # Step 2: Corridor detection - check if we're in a narrow passage
        in_corridor = len(passable_directions) <= 2

        # Step 3: Score each direction
        direction_scores = []
        for direction, target_r, target_c in passable_directions:
            score = 0

            # Base score: unexplored > explored
            if (target_r, target_c) not in state.explored_cells:
                score += 100  # Strong preference for new areas

            # Energy safety: how far is target from charger?
            if nearest_charger:
                dist_to_charger = abs(nearest_charger[0] - target_r) + abs(nearest_charger[1] - target_c)

                if dist_to_charger > max_safe_distance:
                    # Too far from charger - heavily penalize
                    score -= 1000
                else:
                    # Breadth-first: slightly prefer staying closer to charger
                    # This encourages expanding circles rather than deep corridors
                    score -= dist_to_charger * 0.5

            # Corridor behavior: if in corridor, prefer backtracking
            if in_corridor and nearest_charger:
                current_dist = abs(nearest_charger[0] - state.row) + abs(nearest_charger[1] - state.col)
                new_dist = abs(nearest_charger[0] - target_r) + abs(nearest_charger[1] - target_c)

                if new_dist < current_dist:
                    # Moving closer to charger (backtracking) - bonus
                    score += 50
                else:
                    # Going deeper into corridor - penalty
                    score -= 30

            # Small random component to break ties
            score += random.random() * 5

            direction_scores.append((direction, score))

        # Pick highest scoring direction
        best_direction = max(direction_scores, key=lambda x: x[1])[0]
        return best_direction

    def detect_corridor(self, state: HarvestState) -> bool:
        """Detect if agent is in a narrow corridor.

        A corridor is defined as having 2 or fewer passable directions.

        Args:
            state: Current agent state

        Returns:
            True if in corridor, False otherwise
        """
        passable_count = 0
        for direction in ["north", "south", "east", "west"]:
            if self._is_direction_clear(state, direction):
                passable_count += 1

        return passable_count <= 2

    def mark_dead_end(self, state: HarvestState):
        """Mark current position and recent path as dead-end.

        When stuck, marks the current position AND the last 5 positions
        in the movement history. This prevents re-entering the same corridor
        from further back.

        Args:
            state: Current agent state (will be modified)
        """
        current_pos = (state.row, state.col)
        positions_to_mark = [current_pos]

        # Mark last 5 positions in history as dead-end (the corridor leading here)
        for pos in state.position_history[-5:]:
            if pos not in state.dead_end_positions:
                positions_to_mark.append(pos)

        # Add all positions to dead-end set
        for pos in positions_to_mark:
            state.dead_end_positions.add(pos)

        return len(positions_to_mark)

    def _is_direction_clear(self, state: HarvestState, direction: str) -> bool:
        """Check if direction is clear in observation.

        Args:
            state: Current agent state
            direction: Direction to check

        Returns:
            True if clear, False if blocked
        """
        if state.current_obs is None:
            return True  # Optimistic default

        dir_offsets = {"north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1)}
        dr, dc = dir_offsets[direction]
        target_obs_pos = (self._obs_hr + dr, self._obs_wr + dc)

        # Check all tokens at target position
        for tok in state.current_obs.tokens:
            if tok.location == target_obs_pos and tok.feature.name == "tag":
                tag_name = self._tag_names.get(tok.value, "").lower()
                # Block on walls and agents only
                if "wall" in tag_name or tag_name == "agent":
                    return False

        return True

    def find_nearest_frontier_cell(
        self,
        state: HarvestState,
        map_manager: 'MapManager'
    ) -> Optional[tuple[int, int]]:
        """Find nearest explored cell adjacent to UNKNOWN territory.

        CRITICAL: Returns the EXPLORED cell next to the frontier, NOT the UNKNOWN cell.
        This ensures pathfinding works because the target is traversable.

        A "frontier cell" here is an EXPLORED, TRAVERSABLE cell that has at least one
        UNKNOWN neighbor. These cells represent where we can stand and see new territory.

        Args:
            state: Current agent state
            map_manager: MapManager with complete map grid

        Returns:
            Position of nearest frontier cell (explored), or None if none found.
        """
        from .map import MapCellType

        frontier_candidates = []

        # Scan map for frontier cells
        # Adaptive search radius based on map size - larger maps need wider search
        map_dimension = max(state.map_height, state.map_width)
        if map_dimension > 200:
            search_radius = 150  # Large maps (500x500): search 300x300 window
        elif map_dimension > 100:
            search_radius = 75   # Medium maps: search 150x150 window
        else:
            search_radius = 50   # Small maps: search 100x100 window

        start_r = max(0, state.row - search_radius)
        end_r = min(state.map_height, state.row + search_radius + 1)
        start_c = max(0, state.col - search_radius)
        end_c = min(state.map_width, state.col + search_radius + 1)

        for r in range(start_r, end_r):
            for c in range(start_c, end_c):
                # FIXED: Must be EXPLORED and TRAVERSABLE (not unknown, not wall)
                cell_type = map_manager.grid[r][c]
                if cell_type in (MapCellType.UNKNOWN, MapCellType.WALL, MapCellType.DEAD_END):
                    continue

                # Check if adjacent to any UNKNOWN cell (this is the frontier!)
                is_frontier = False
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < state.map_height and 0 <= nc < state.map_width:
                        neighbor_type = map_manager.grid[nr][nc]
                        # Adjacent to unexplored area
                        if neighbor_type == MapCellType.UNKNOWN:
                            is_frontier = True
                            break

                if is_frontier:
                    frontier_candidates.append((r, c))

        if not frontier_candidates:
            return None

        # Return nearest frontier cell using Manhattan distance
        current = (state.row, state.col)
        return min(
            frontier_candidates,
            key=lambda pos: abs(pos[0] - current[0]) + abs(pos[1] - current[1])
        )
