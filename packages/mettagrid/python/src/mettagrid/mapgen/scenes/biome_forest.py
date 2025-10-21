import numpy as np

from mettagrid.config.config import Config
from mettagrid.mapgen.scene import Scene


class BiomeForestParams(Config):
    clumpiness: int = 3  # number of growth passes
    seed_prob: float = 0.04  # initial random seeds density
    growth_prob: float = 0.55  # probability to grow into empty if enough neighbors
    neighbor_threshold: int = 3  # forest spreads when >= this many forest neighbors
    dither_edges: bool = True  # Add organic edge noise
    dither_prob: float = 0.15  # Probability to flip edge cells
    dither_depth: int = 5  # How many cells deep to consider as edge zone


class BiomeForest(Scene[BiomeForestParams]):
    """
    Cellular automata style forest: clumpy organic shapes.
    Walls are "trees"; empty is passable ground.
    """

    def render(self):
        grid = self.grid
        H, W = self.height, self.width
        p = self.params

        forest = np.zeros((H, W), dtype=np.uint8)
        # Seed
        seeds = self.rng.random((H, W)) < p.seed_prob
        forest[seeds] = 1

        # Grow
        for _ in range(max(0, int(p.clumpiness))):
            nb = (
                np.pad(forest, 1)[0:H, 1 : W + 1]
                + np.pad(forest, 1)[2 : H + 2, 1 : W + 1]
                + np.pad(forest, 1)[1 : H + 1, 0:W]
                + np.pad(forest, 1)[1 : H + 1, 2 : W + 2]
                + np.pad(forest, 1)[0:H, 0:W]
                + np.pad(forest, 1)[0:H, 2 : W + 2]
                + np.pad(forest, 1)[2 : H + 2, 0:W]
                + np.pad(forest, 1)[2 : H + 2, 2 : W + 2]
            )
            grow = (nb >= p.neighbor_threshold) & (self.rng.random((H, W)) < p.growth_prob)
            forest = np.where(grow | (forest == 1), 1, 0).astype(np.uint8)

        # Stamp into grid
        grid[forest == 1] = "wall"

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
