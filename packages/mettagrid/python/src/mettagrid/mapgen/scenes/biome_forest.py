import numpy as np

from mettagrid.mapgen.scene import Scene, SceneConfig
from mettagrid.mapgen.scenes.dither import dither_edges


class BiomeForestConfig(SceneConfig):
    clumpiness: int = 2  # number of growth passes
    seed_prob: float = 0.03  # initial random seeds density
    growth_prob: float = 0.5  # probability to grow into empty if enough neighbors
    neighbor_threshold: int = 3  # forest spreads when >= this many forest neighbors
    dither_edges: bool = True  # Add organic edge noise
    dither_prob: float = 0.15  # Probability to flip edge cells
    dither_depth: int = 5  # How many cells deep to consider as edge zone


class BiomeForest(Scene[BiomeForestConfig]):
    """
    Cellular automata style forest: clumpy organic shapes.
    Walls are "trees"; empty is passable ground.
    """

    def render(self):
        grid = self.grid
        H, W = self.height, self.width
        p = self.config

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
            dither_edges(grid, prob=p.dither_prob, depth=p.dither_depth, rng=self.rng)
