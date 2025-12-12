"""
Fast pathfinding utilities with caching.

This module provides optimized pathfinding functions that avoid
repeated allocations and use simpler data structures.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import CellType


class PathCache:
    """Cache for pathfinding computations with distance maps for stations."""

    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width
        # Distance maps for each station: station_name -> (distances, came_from)
        # distances[r][c] = distance from station, or -1 if unreachable
        # came_from[(r,c)] = previous position for path reconstruction
        self._station_maps: dict[str, tuple[list[list[int]], dict[tuple[int, int], tuple[int, int] | None]]] = {}

    def compute_distance_map(
        self,
        station_name: str,
        station_pos: tuple[int, int],
        occupancy: list[list[int]],
        cell_type: type[CellType],
        agent_occupancy: set[tuple[int, int]],
    ) -> None:
        """Pre-compute distance map from station using BFS.

        Creates a distance field and came_from map for efficient path queries.
        """
        distances = [[-1] * self.width for _ in range(self.height)]
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {}

        # BFS from all adjacent walkable cells of the station
        queue: deque[tuple[int, int]] = deque()
        sr, sc = station_pos

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = sr + dr, sc + dc
            if 0 <= nr < self.height and 0 <= nc < self.width:
                if occupancy[nr][nc] == cell_type.FREE.value:
                    distances[nr][nc] = 0
                    came_from[(nr, nc)] = None  # These are the goal positions
                    queue.append((nr, nc))

        # BFS to compute distances
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    if distances[nr][nc] == -1 and occupancy[nr][nc] == cell_type.FREE.value:
                        distances[nr][nc] = distances[r][c] + 1
                        came_from[(nr, nc)] = (r, c)
                        queue.append((nr, nc))

        self._station_maps[station_name] = (distances, came_from)

    def get_path_to_station(
        self,
        station_name: str,
        start: tuple[int, int],
        occupancy: list[list[int]],
        cell_type: type[CellType],
        agent_occupancy: set[tuple[int, int]],
    ) -> list[tuple[int, int]] | None:
        """Get cached path to station if available.

        Returns path from start to adjacent cell of station, or None if no cache.
        """
        if station_name not in self._station_maps:
            return None

        distances, came_from = self._station_maps[station_name]
        sr, sc = start

        # Check if start is reachable
        if sr < 0 or sr >= self.height or sc < 0 or sc >= self.width:
            return None
        if distances[sr][sc] == -1:
            return None

        # Reconstruct path by following decreasing distances
        path: list[tuple[int, int]] = []
        current = start

        while came_from.get(current) is not None:
            next_pos = came_from[current]
            if next_pos is None:
                break
            # Check if path is still valid (no agent blocking)
            if next_pos in agent_occupancy:
                return None  # Path blocked, need fresh BFS
            path.append(next_pos)
            current = next_pos

        return path if path else None

    def shortest_path_fast(
        self,
        start: tuple[int, int],
        goals: list[tuple[int, int]],
        occupancy: list[list[int]],
        cell_type: type[CellType],
        agent_occupancy: set[tuple[int, int]],
        allow_goal_block: bool,
    ) -> list[tuple[int, int]]:
        """Find shortest path using BFS.

        Similar to regular shortest_path but uses raw data structures.
        """
        if not goals:
            return []

        goal_set = set(goals)
        queue: deque[tuple[int, int]] = deque([start])
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

        def is_walkable(r: int, c: int) -> bool:
            if not (0 <= r < self.height and 0 <= c < self.width):
                return False
            if (r, c) in goal_set and allow_goal_block:
                return True
            if (r, c) in agent_occupancy:
                return False
            return occupancy[r][c] == cell_type.FREE.value

        while queue:
            current = queue.popleft()
            if current in goal_set:
                # Reconstruct path
                path: list[tuple[int, int]] = []
                while came_from[current] is not None:
                    path.append(current)
                    prev = came_from[current]
                    assert prev is not None
                    current = prev
                path.reverse()
                return path

            r, c = current
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in came_from and is_walkable(nr, nc):
                    came_from[(nr, nc)] = current
                    queue.append((nr, nc))

        return []


def compute_goal_cells_fast(
    occupancy: list[list[int]],
    map_height: int,
    map_width: int,
    agent_occupancy: set[tuple[int, int]],
    target: tuple[int, int],
    reach_adjacent: bool,
    cell_type: type[CellType],
) -> list[tuple[int, int]]:
    """
    Compute the set of goal cells for pathfinding.

    Faster version that takes raw data instead of state object.
    """
    if not reach_adjacent:
        return [target]

    goals: list[tuple[int, int]] = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = target[0] + dr, target[1] + dc
        if 0 <= nr < map_height and 0 <= nc < map_width:
            # Check if traversable (free and no agent)
            if occupancy[nr][nc] == cell_type.FREE.value and (nr, nc) not in agent_occupancy:
                goals.append((nr, nc))

    # If no adjacent traversable tiles are known yet, allow exploring toward unknown ones
    if not goals:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = target[0] + dr, target[1] + dc
            if 0 <= nr < map_height and 0 <= nc < map_width:
                if occupancy[nr][nc] != cell_type.OBSTACLE.value:
                    goals.append((nr, nc))

    return goals
