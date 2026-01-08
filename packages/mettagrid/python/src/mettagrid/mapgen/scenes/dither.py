from __future__ import annotations

import numpy as np


def dither_edges(
    grid: np.ndarray,
    prob: float,
    depth: int,
    rng: np.random.Generator,
) -> None:
    """
    Vectorized edge dithering for organic biome boundaries.

    Adds wall/empty noise near boundaries in O(H·W·depth) time (no Python
    triple loops). Mutates `grid` in place.
    """
    if depth <= 0 or prob <= 0.0:
        return

    wall = grid == "wall"
    empty = ~wall

    # Edge cells: 8-neighbor (Chebyshev) difference.
    def _expand(mask: np.ndarray) -> np.ndarray:
        up = np.zeros_like(mask, dtype=bool)
        down = np.zeros_like(mask, dtype=bool)
        left = np.zeros_like(mask, dtype=bool)
        right = np.zeros_like(mask, dtype=bool)
        up[:-1] = mask[1:]
        down[1:] = mask[:-1]
        left[:, :-1] = mask[:, 1:]
        right[:, 1:] = mask[:, :-1]
        up_left = np.zeros_like(mask, dtype=bool)
        up_right = np.zeros_like(mask, dtype=bool)
        down_left = np.zeros_like(mask, dtype=bool)
        down_right = np.zeros_like(mask, dtype=bool)
        up_left[:-1, :-1] = mask[1:, 1:]
        up_right[:-1, 1:] = mask[1:, :-1]
        down_left[1:, :-1] = mask[:-1, 1:]
        down_right[1:, 1:] = mask[:-1, :-1]
        return up | down | left | right | up_left | up_right | down_left | down_right

    # Boundary cells = any cell adjacent (8-neighbor) to opposite type.
    boundary = (_expand(wall) & empty) | (_expand(empty) & wall)
    frontier = boundary
    dist = np.full(grid.shape, np.inf, dtype=np.float32)
    dist[frontier] = 0.0
    seen = frontier.copy()
    current_depth = 0

    while current_depth < depth and frontier.any():
        current_depth += 1
        frontier = _expand(frontier) & (~seen)
        if not frontier.any():
            break
        dist[frontier] = current_depth
        seen |= frontier

    reachable = dist <= depth
    if not reachable.any():
        return

    # Exclude border band (match prior branch behavior).
    reachable[:depth, :] = False
    reachable[-depth:, :] = False
    reachable[:, :depth] = False
    reachable[:, -depth:] = False

    # Ensure nearest edge distance is 1 (match legacy behavior: distance=1 => prob).
    effective_dist = np.maximum(1.0, dist)
    edge_prob = prob * (depth - effective_dist + 1) / depth
    flips = (rng.random(grid.shape) < edge_prob) & reachable

    grid[flips & wall] = "empty"
    grid[flips & empty] = "wall"
