"""
Utility functions for scripted agents.

Pure/stateless helper functions that can be reused across different agents.
"""

from __future__ import annotations


def manhattan_distance(pos1: tuple[int, int], pos2: tuple[int, int]) -> int:
    """Calculate Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def is_adjacent(pos1: tuple[int, int], pos2: tuple[int, int]) -> bool:
    """Check if two positions are adjacent (4-way cardinal directions)."""
    dr = abs(pos1[0] - pos2[0])
    dc = abs(pos1[1] - pos2[1])
    return (dr == 1 and dc == 0) or (dr == 0 and dc == 1)


def is_within_bounds(pos: tuple[int, int], map_height: int, map_width: int) -> bool:
    """Check if a position is within map bounds."""
    r, c = pos
    return 0 <= r < map_height and 0 <= c < map_width


def get_cardinal_neighbors(pos: tuple[int, int]) -> list[tuple[int, int]]:
    """Get all 4-way cardinal neighbor positions."""
    r, c = pos
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def is_wall(obj_name: str) -> bool:
    """Check if an object name represents a wall or obstacle."""
    return "wall" in obj_name or "#" in obj_name or obj_name in {"wall", "obstacle"}


def is_floor(obj_name: str) -> bool:
    """Check if an object name represents floor (passable empty space)."""
    # Environment returns empty string for empty cells
    return obj_name in {"floor", ""}


def is_station(obj_name: str, station: str) -> bool:
    """Check if an object name contains a specific station type."""
    return station in obj_name


def position_to_direction(from_pos: tuple[int, int], to_pos: tuple[int, int]) -> str | None:
    """
    Convert adjacent positions to a cardinal direction name.

    Returns: "north", "south", "east", "west", or None if not adjacent.
    """
    dr = to_pos[0] - from_pos[0]
    dc = to_pos[1] - from_pos[1]

    if dr == -1 and dc == 0:
        return "north"
    elif dr == 1 and dc == 0:
        return "south"
    elif dr == 0 and dc == 1:
        return "east"
    elif dr == 0 and dc == -1:
        return "west"
    return None
