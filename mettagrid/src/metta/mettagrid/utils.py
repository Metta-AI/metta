"""Utility functions for MettaGrid."""

import random
from typing import Dict, List, Optional

import numpy as np

from metta.mettagrid.level_builder import LevelMap


def make_level_map(
    width: int,
    height: int,
    objects: Optional[Dict[str, int]] = None,
    num_agents: int = 1,
    border_width: int = 1,
    seed: Optional[int] = None,
    labels: Optional[List[str]] = None,
) -> LevelMap:
    """
    Create a random level map.

    Args:
        width: Width of the map
        height: Height of the map
        objects: Dictionary mapping object names to counts (e.g., {"altar": 2, "mine_red": 3})
        num_agents: Number of agents to place
        border_width: Width of wall border around the map
        seed: Random seed for reproducibility
        labels: Optional list of labels for the map

    Returns:
        A LevelMap object with a 2D numpy array grid and labels
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Initialize empty map with enough space for agent names
    level_map = np.full((height, width), "empty", dtype="<U20")

    # Add border walls if requested
    if border_width > 0:
        # Top and bottom borders
        level_map[:border_width, :] = "wall"
        level_map[-border_width:, :] = "wall"
        # Left and right borders
        level_map[:, :border_width] = "wall"
        level_map[:, -border_width:] = "wall"

    # Get available positions (not wall)
    available_positions = []
    for y in range(height):
        for x in range(width):
            if level_map[y, x] == "empty":
                available_positions.append((y, x))

    # Shuffle available positions
    random.shuffle(available_positions)

    # Place agents - use "agent.agent" for the default agent group
    for _i in range(min(num_agents, len(available_positions))):
        y, x = available_positions.pop()
        level_map[y, x] = "agent.agent"

    # Place objects
    if objects:
        for obj_name, count in objects.items():
            for _ in range(min(count, len(available_positions))):
                if not available_positions:
                    break
                y, x = available_positions.pop()
                level_map[y, x] = obj_name

    return LevelMap(grid=level_map, labels=labels or [])
