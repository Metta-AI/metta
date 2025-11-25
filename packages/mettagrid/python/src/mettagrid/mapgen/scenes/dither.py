"""Vectorized edge dithering for organic biome boundaries.

This module provides efficient NumPy-based edge dithering that replaces
the O(n² × d²) pure Python implementation with O(n² × d) vectorized ops.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator


def dither_edges(
    grid: np.ndarray,
    prob: float,
    depth: int,
    rng: Generator,
) -> None:
    """Add organic noise to edges between wall and empty cells.

    This vectorized implementation is ~100x faster than the naive nested loop
    approach for typical grid sizes and depth values.

    Args:
        grid: The grid array (modified in-place). Cells are "wall" or "empty".
        prob: Base probability to flip edge cells (0-1).
        depth: How many cells deep to consider as edge zone.
        rng: NumPy random generator for reproducibility.
    """
    if prob <= 0 or depth <= 0:
        return

    H, W = grid.shape
    if H <= 2 * depth or W <= 2 * depth:
        return

    # Create binary mask: True for walls
    is_wall = grid == "wall"

    # Compute distance to nearest opposite-type cell using iterative dilation
    # This replaces scipy.ndimage.distance_transform with pure NumPy
    dist_to_edge = _compute_edge_distance(is_wall, depth)

    # Create probability mask based on distance
    # Closer to edge = higher probability
    # At distance 1: prob, at distance depth: prob/depth
    valid_mask = (dist_to_edge > 0) & (dist_to_edge <= depth)

    # Exclude border cells
    valid_mask[:depth, :] = False
    valid_mask[-depth:, :] = False
    valid_mask[:, :depth] = False
    valid_mask[:, -depth:] = False

    if not np.any(valid_mask):
        return

    # Compute edge probability: higher when closer to edge
    edge_prob = np.zeros((H, W), dtype=np.float32)
    edge_prob[valid_mask] = prob * (depth - dist_to_edge[valid_mask] + 1) / depth

    # Generate all random values at once (vectorized)
    random_vals = rng.random((H, W), dtype=np.float32)
    flip_mask = (random_vals < edge_prob) & valid_mask

    # Apply flips: wall -> empty, empty -> wall
    flip_to_empty = flip_mask & is_wall
    flip_to_wall = flip_mask & ~is_wall

    grid[flip_to_empty] = "empty"
    grid[flip_to_wall] = "wall"


def _compute_edge_distance(is_wall: np.ndarray, max_depth: int) -> np.ndarray:
    """Compute Chebyshev distance to nearest cell of opposite type.

    Uses iterative erosion/dilation which is O(depth) per cell rather than
    checking all neighbors in a (2*depth+1)² window.

    Returns array where value at (y,x) is the Chebyshev distance to the
    nearest cell of opposite type (capped at max_depth + 1).
    """
    H, W = is_wall.shape

    # Initialize distances to max_depth + 1 (means "far from edge")
    dist = np.full((H, W), max_depth + 1, dtype=np.uint8)

    # Find initial edges: cells adjacent to opposite type
    # We'll grow the edge zone iteratively
    is_wall_padded = np.pad(is_wall, 1, mode="edge")

    for d in range(1, max_depth + 1):
        # For each cell, check if any neighbor within distance d is opposite type
        # We only need to check the "shell" at exactly distance d

        # Use the 8-connected neighborhood check via max-pooling style operation
        # A cell is at distance d if it wasn't marked at distance < d
        # and has an opposite-type cell at exactly distance d

        if d == 1:
            # Direct neighbors (8-connected)
            has_opposite_neighbor = np.zeros((H, W), dtype=bool)
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    shifted = is_wall_padded[1 + dy : H + 1 + dy, 1 + dx : W + 1 + dx]
                    has_opposite_neighbor |= shifted != is_wall
            dist[has_opposite_neighbor] = 1
        else:
            # For d > 1, we need cells that:
            # - Don't have distance < d yet
            # - Have a neighbor at distance d-1
            unmarked = dist > d
            if not np.any(unmarked):
                break

            # Check if any cell at distance d-1 is in our Chebyshev neighborhood
            at_prev_dist = dist == d - 1
            at_prev_padded = np.pad(at_prev_dist, 1, mode="constant", constant_values=False)

            has_prev_neighbor = np.zeros((H, W), dtype=bool)
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    shifted = at_prev_padded[1 + dy : H + 1 + dy, 1 + dx : W + 1 + dx]
                    has_prev_neighbor |= shifted

            new_at_d = unmarked & has_prev_neighbor
            dist[new_at_d] = d

    return dist
