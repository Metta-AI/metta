import numpy as np

from mettagrid.config.config import Config
from mettagrid.mapgen.scene import Scene


class BiomeForestParams(Config):
    clumpiness: int = 3  # number of growth passes
    seed_prob: float = 0.04  # initial random seeds density
    growth_prob: float = 0.55  # probability to grow into empty if enough neighbors
    neighbor_threshold: int = 3  # forest spreads when >= this many forest neighbors


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
