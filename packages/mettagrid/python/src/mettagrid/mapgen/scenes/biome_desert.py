import numpy as np

from mettagrid.mapgen.scene import Scene, SceneConfig
from mettagrid.mapgen.scenes.dither import dither_edges


class BiomeDesertConfig(SceneConfig):
    dune_period: int = 8  # distance between dune ridges
    ridge_width: int = 1  # ridge thickness in cells
    angle: float = np.pi / 4  # radians; 0 == vertical stripes, pi/4 diagonal, etc.
    noise_prob: float = 0.1  # pepper dunes with small gaps
    dither_edges: bool = True  # Add organic edge noise
    dither_prob: float = 0.15  # Probability to flip edge cells
    dither_depth: int = 5  # How many cells deep to consider as edge zone


class BiomeDesert(Scene[BiomeDesertConfig]):
    """
    Striated dunes: parallel ridges with occasional gaps.
    Walls are dunes; empty is sand path.
    """

    def render(self):
        grid = self.grid
        H, W = self.height, self.width
        p = self.config

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
            dither_edges(grid, prob=p.dither_prob, depth=p.dither_depth, rng=self.rng)
