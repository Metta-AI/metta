import numpy as np

from mettagrid.config.config import Config
from mettagrid.mapgen.scene import Scene


class BiomeDesertParams(Config):
    dune_period: int = 8  # distance between dune ridges
    ridge_width: int = 2  # ridge thickness in cells
    angle: float = 0.0  # radians; 0 == vertical stripes, pi/4 diagonal, etc.
    noise_prob: float = 0.3  # pepper dunes with small gaps
    dither_edges: bool = True  # Add organic edge noise
    dither_prob: float = 0.15  # Probability to flip edge cells
    dither_depth: int = 5  # How many cells deep to consider as edge zone


class BiomeDesert(Scene[BiomeDesertParams]):
    """
    Striated dunes: parallel ridges with occasional gaps.
    Walls are dunes; empty is sand path.
    """

    def render(self):
        grid = self.grid
        H, W = self.height, self.width
        p = self.params

        period = max(2, int(p.dune_period))
        width = max(1, int(p.ridge_width))
        theta = float(p.angle)

        # Build coordinate field rotated by theta
        ys, xs = np.indices((H, W))
        # Rotate coordinates
        xr = xs * np.cos(theta) + ys * np.sin(theta)
        mask = (xr % period) < width

        dunes = mask.astype(np.uint8)
        # Pepper with noise holes, increase probability to 0.3
        holes = self.rng.random((H, W)) < p.noise_prob
        dunes = np.where(holes, 0, dunes)

        grid[dunes == 1] = "wall"

        # Apply edge dithering for organic look
        if p.dither_edges:
            self._dither_edges(grid, p.dither_prob)

    def _dither_edges(self, grid, prob: float):
        """Add organic noise to edges between wall and empty cells."""
        H, W = grid.shape
        depth = self.params.dither_depth

        # Find edges: cells within 'depth' distance of opposite type
        for y in range(depth, H - depth):
            for x in range(depth, W - depth):
                current = grid[y, x]

                # Find distance to nearest opposite type cell
                min_dist = depth + 1
                for dy in range(-depth, depth + 1):
                    for dx in range(-depth, depth + 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W:
                            if grid[ny, nx] != current:
                                dist = max(abs(dy), abs(dx))  # Chebyshev distance
                                min_dist = min(min_dist, dist)

                # Only dither cells near edges (within depth)
                if min_dist <= depth:
                    # Probability increases as we get closer to edge
                    # At distance 1: prob, at distance depth: prob/depth
                    edge_prob = prob * (depth - min_dist + 1) / depth
                    if self.rng.random() < edge_prob:
                        grid[y, x] = "empty" if current == "wall" else "wall"
