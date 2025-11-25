import numpy as np
import scipy.ndimage


def dither_edges(grid, prob: float, depth: int, rng: np.random.Generator):
    """
    Add organic noise to edges between wall and empty cells.
    Optimized version using scipy.ndimage.distance_transform_cdt.
    """
    if prob <= 0 or depth <= 0:
        return

    H, W = grid.shape

    # Create binary mask: 1 for wall, 0 for empty
    # We assume anything not "wall" is treated as empty for distance calculation purposes
    # consistent with the original logic checking `grid[ny, nx] != current`
    is_wall = grid == "wall"

    # Distance to nearest empty cell (for wall pixels)
    # distance_transform_cdt calculates distance to nearest ZERO
    # So we pass is_wall (walls are 1, empty is 0).
    dist_to_empty = scipy.ndimage.distance_transform_cdt(is_wall, metric="chessboard")

    # Distance to nearest wall cell (for empty pixels)
    # We pass ~is_wall (empty is 1, walls are 0).
    dist_to_wall = scipy.ndimage.distance_transform_cdt(~is_wall, metric="chessboard")

    # Combine to get "distance to nearest edge"
    # For a wall pixel, dist is dist_to_empty. For empty pixel, dist is dist_to_wall.
    # Note: distance_transform_cdt returns 1 for pixels immediately adjacent to 0.
    dist_to_edge = np.where(is_wall, dist_to_empty, dist_to_wall)

    # Mask for cells within depth
    # Original logic: `if min_dist <= depth`
    mask = dist_to_edge <= depth

    if not np.any(mask):
        return

    # Calculate probabilities
    # Original logic: edge_prob = prob * (depth - min_dist + 1) / depth
    edge_probs = prob * (depth - dist_to_edge + 1) / depth

    # Generate random flips
    random_vals = rng.random(grid.shape)
    should_flip = mask & (random_vals < edge_probs)

    # Apply flips
    # We only flip if it's wall or empty. If there are other things, we might need to be careful.
    # But original logic: `grid[y, x] = "empty" if current == "wall" else "wall"`
    # implies it only flips between wall and empty.

    # Optimization: Use boolean indexing
    flips = should_flip
    if not np.any(flips):
        return

    # We need to flip "wall" to "empty" and "empty" (or anything else) to "wall"
    # Note: The original logic `grid[y, x] = "empty" if current == "wall" else "wall"`
    # means if it was "agent" it becomes "wall".

    current_walls = grid == "wall"
    grid[flips & current_walls] = "empty"
    grid[flips & ~current_walls] = "wall"
