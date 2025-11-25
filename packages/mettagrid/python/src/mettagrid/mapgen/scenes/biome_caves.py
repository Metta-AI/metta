import numpy as np

from mettagrid.mapgen.scene import Scene, SceneConfig


class BiomeCavesConfig(SceneConfig):
    fill_prob: float = 0.45  # initial rock density
    steps: int = 4  # cellular automata smoothing steps
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
            self._dither_edges(grid, p.dither_prob)

    def _dither_edges(self, grid, prob: float):
        """Add organic noise to edges between wall and empty cells."""
        H, W = grid.shape
        depth = self.config.dither_depth

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
