from __future__ import annotations

import numpy as np


def vectorized_dither_edges(
    grid: np.ndarray,
    prob: float,
    depth: int,
    rng: np.random.Generator,
) -> None:
    """
    Add organic edge noise without per-cell Python loops.

    Algorithm:
    1) Identify edge cells (wall/empty boundary).
    2) Iteratively dilate the edge frontier up to `depth`, recording the
       Chebyshev distance to the nearest edge for all cells reached.
    3) Flip cells with probability scaled by distance to the boundary.
    """
    if depth <= 0 or prob <= 0.0:
        return

    wall = grid == "wall"
    empty = ~wall

    # Edge = any cell whose 4-neighborhood differs in type.
    def _expand(mask: np.ndarray) -> np.ndarray:
        up = np.zeros_like(mask, dtype=bool)
        down = np.zeros_like(mask, dtype=bool)
        left = np.zeros_like(mask, dtype=bool)
        right = np.zeros_like(mask, dtype=bool)
        up[:-1] = mask[1:]
        down[1:] = mask[:-1]
        left[:, :-1] = mask[:, 1:]
        right[:, 1:] = mask[:, :-1]
        return up | down | left | right

    neighbor_diff = _expand(wall) ^ _expand(empty)
    edge_frontier = neighbor_diff

    dist = np.full(grid.shape, np.inf, dtype=np.float32)
    dist[edge_frontier] = 0.0

    seen = edge_frontier.copy()
    current_depth = 0
    frontier = edge_frontier

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

    edge_prob = prob * (depth - dist + 1) / depth
    flips = (rng.random(grid.shape) < edge_prob) & reachable

    # Flip wall<->empty where requested.
    grid[flips & wall] = "empty"
    grid[flips & empty] = "wall"
