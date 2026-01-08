import numpy as np

from mettagrid.mapgen.scene import Scene, SceneConfig
from mettagrid.mapgen.scenes.dither import dither_edges


class BiomeCavesConfig(SceneConfig):
    fill_prob: float = 0.4  # initial rock density
    steps: int = 3  # cellular automata smoothing steps
    birth_limit: int = 5
    death_limit: int = 3
    dither_edges: bool = True  # Add organic edge noise
    dither_prob: float = 0.15  # Probability to flip edge cells
    dither_depth: int = 5  # How many cells deep to consider as edge zone


class BiomeCaves(Scene[BiomeCavesConfig]):
    """
    Classic cellular-automata caves (rooms connected via erosion-like smoothing).
    Walls are rock; empty is cave passage.
    """

    def render(self):
        grid = self.grid
        H, W = self.height, self.width
        p = self.config

        rock = (self.rng.random((H, W)) < p.fill_prob).astype(np.uint8)

        def count_neighbors(a: np.ndarray) -> np.ndarray:
            a_p = np.pad(a, 1, mode="constant", constant_values=1)  # treat outside as rock
            nb = (
                a_p[0:H, 1 : W + 1]
                + a_p[2 : H + 2, 1 : W + 1]
                + a_p[1 : H + 1, 0:W]
                + a_p[1 : H + 1, 2 : W + 2]
                + a_p[0:H, 0:W]
                + a_p[0:H, 2 : W + 2]
                + a_p[2 : H + 2, 0:W]
                + a_p[2 : H + 2, 2 : W + 2]
            )
            return nb

        for _ in range(max(0, int(p.steps))):
            nb = count_neighbors(rock)
            birth = nb > p.birth_limit
            death = nb < p.death_limit
            rock = np.where(birth | ((~death) & (rock == 1)), 1, 0).astype(np.uint8)

        grid[rock == 1] = "wall"

        # Apply edge dithering for organic look
        if p.dither_edges:
            dither_edges(grid, prob=p.dither_prob, depth=p.dither_depth, rng=self.rng)
