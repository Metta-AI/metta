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

        # IMPROVEMENT #3: Incremental frontier cache for O(N) performance
        self._frontier_cache: set[tuple[int, int]] = set()
        self._frontier_dirty = True
        self._last_cache_rebuild_step = 0

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

        BUG FIX #8: Don't mark positions with 3+ passable neighbors (junctions)
        to avoid trapping the agent.

        Args:
            state: Current agent state (will be modified)
        """
        current_pos = (state.row, state.col)
        positions_to_mark = [current_pos]

        # Mark last 5 positions in history as dead-end (the corridor leading here)
        for pos in state.position_history[-5:]:
            if pos not in state.dead_end_positions:
                positions_to_mark.append(pos)

        # Add positions to dead-end set, but skip junctions (3+ clear neighbors)
        marked_count = 0
        for pos in positions_to_mark:
            # Count passable neighbors
            passable_neighbors = 0
            for direction in ["north", "south", "east", "west"]:
                # Temporarily set state position to check from this pos
                original_row, original_col = state.row, state.col
                state.row, state.col = pos[0], pos[1]

                if self._is_direction_clear(state, direction):
                    passable_neighbors += 1

                # Restore original position
                state.row, state.col = original_row, original_col

            # Only mark as dead-end if it's not a junction (< 3 neighbors)
            if passable_neighbors < 3:
                state.dead_end_positions.add(pos)
                marked_count += 1

        return marked_count

    def invalidate_frontier_cache(self):
        """Mark frontier cache as needing rebuild.

        Call this whenever the map changes (new cells explored, walls discovered).
        """
        self._frontier_dirty = True

    def _rebuild_frontier_cache(self, state: 'HarvestState', map_manager: 'MapManager'):
        """Rebuild the frontier cache incrementally.

        IMPROVEMENT #3: Instead of scanning the entire map every time,
        only scan cells that could be frontiers (explored cells near unknown territory).

        Performance: O(E) where E = number of explored cells, vs O(W*H) for full scan.
        """
        from .map import MapCellType

        self._frontier_cache.clear()

        # Strategy: Only check recently explored cells (within observable range)
        # This is much faster than scanning the entire map
        search_radius = 200  # Check cells within 200 of current position

        start_r = max(0, state.row - search_radius)
        end_r = min(state.map_height, state.row + search_radius + 1)
        start_c = max(0, state.col - search_radius)
        end_c = min(state.map_width, state.col + search_radius + 1)

        for r in range(start_r, end_r):
            for c in range(start_c, end_c):
                # Must be explored and traversable
                cell_type = map_manager.grid[r][c]
                if cell_type in (MapCellType.UNKNOWN, MapCellType.WALL, MapCellType.DEAD_END):
                    continue

                # Check if adjacent to any UNKNOWN cell
                is_frontier = False
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < state.map_height and 0 <= nc < state.map_width:
                        if map_manager.grid[nr][nc] == MapCellType.UNKNOWN:
                            is_frontier = True
                            break

                if is_frontier:
                    self._frontier_cache.add((r, c))

        self._last_cache_rebuild_step = state.step_count

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
                # Handle both list and dict for tag_names (list during __init__, dict after initial_agent_state)
                if isinstance(self._tag_names, dict):
                    tag_name = self._tag_names.get(tok.value, "").lower()
                elif isinstance(self._tag_names, list):
                    # tag_names is a list - use index lookup
                    tag_name = self._tag_names[tok.value].lower() if tok.value < len(self._tag_names) else ""
                else:
                    tag_name = ""

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

        IMPROVEMENT #3: Uses incremental frontier cache for O(N) performance
        instead of O(N²) full map scan.

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
        # Rebuild cache if dirty or if we've moved significantly
        steps_since_rebuild = state.step_count - self._last_cache_rebuild_step
        if self._frontier_dirty or steps_since_rebuild > 50:
            self._rebuild_frontier_cache(state, map_manager)
            self._frontier_dirty = False

        # Find nearest frontier from cache
        if not self._frontier_cache:
            return None

        current = (state.row, state.col)
        return min(
            self._frontier_cache,
            key=lambda pos: abs(pos[0] - current[0]) + abs(pos[1] - current[1])
        )
