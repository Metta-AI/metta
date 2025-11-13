"""
Pathfinding utilities for scripted agents.

This module contains A* pathfinding implementation and related utilities
for navigating the grid world.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cogames.policy.scripted_agent.types import CellType, SimpleAgentState


def compute_goal_cells(
    state: SimpleAgentState, target: tuple[int, int], reach_adjacent: bool, cell_type: type[CellType]
) -> list[tuple[int, int]]:
    """
    Compute the set of goal cells for pathfinding.
    """
    if not reach_adjacent:
        return [target]

    goals: list[tuple[int, int]] = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = target[0] + dr, target[1] + dc
        if is_traversable(state, nr, nc, cell_type):
            goals.append((nr, nc))

    # If no adjacent traversable tiles are known yet, allow exploring toward unknown ones
    if not goals:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = target[0] + dr, target[1] + dc
            if is_within_bounds(state, nr, nc) and state.occupancy[nr][nc] != cell_type.OBSTACLE.value:
                goals.append((nr, nc))
    return goals


def shortest_path(
    state: SimpleAgentState,
    start: tuple[int, int],
    goals: list[tuple[int, int]],
    allow_goal_block: bool,
    cell_type: type[CellType],
) -> list[tuple[int, int]]:
    """
    Find shortest path from start to any goal using BFS.
    """
    goal_set = set(goals)
    queue: deque[tuple[int, int]] = deque([start])
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

    def walkable(r: int, c: int) -> bool:
        if (r, c) in goal_set and allow_goal_block:
            return True
        return is_traversable(state, r, c, cell_type)

    while queue:
        current = queue.popleft()
        if current in goal_set:
            return reconstruct_path(came_from, current)

        for nr, nc in get_neighbors(state, current):
            if (nr, nc) not in came_from and walkable(nr, nc):
                came_from[(nr, nc)] = current
                queue.append((nr, nc))

    return []


def reconstruct_path(
    came_from: dict[tuple[int, int], tuple[int, int] | None],
    current: tuple[int, int],
) -> list[tuple[int, int]]:
    """
    Reconstruct path from BFS came_from dict.
    """
    path: list[tuple[int, int]] = []
    while came_from[current] is not None:
        path.append(current)
        prev = came_from[current]
        assert prev is not None  # Loop condition ensures this
        current = prev
    path.reverse()
    return path


def get_neighbors(state: SimpleAgentState, pos: tuple[int, int]) -> list[tuple[int, int]]:
    """
    Get valid neighboring positions (4-connected grid).
    """
    r, c = pos
    candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [(nr, nc) for nr, nc in candidates if is_within_bounds(state, nr, nc)]


def is_within_bounds(state: SimpleAgentState, r: int, c: int) -> bool:
    """
    Check if position is within map bounds.
    """
    return 0 <= r < state.map_height and 0 <= c < state.map_width


def is_passable(state: SimpleAgentState, r: int, c: int, cell_type: type[CellType]) -> bool:
    """
    Check if a cell is passable (not an obstacle).
    """
    if not is_within_bounds(state, r, c):
        return False
    return is_traversable(state, r, c, cell_type)


def is_traversable(state: SimpleAgentState, r: int, c: int, cell_type: type[CellType]) -> bool:
    """
    Check if a cell is traversable (free and no agent there).
    """
    if not is_within_bounds(state, r, c):
        return False
    # Don't walk through other agents
    if (r, c) in state.agent_occupancy:
        return False
    cell = state.occupancy[r][c]
    # Only traverse cells we KNOW are free, not unknown cells
    return cell == cell_type.FREE.value
