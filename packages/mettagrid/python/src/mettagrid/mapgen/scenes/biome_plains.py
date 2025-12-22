import numpy as np

from mettagrid.mapgen.scene import Scene, SceneConfig


class BiomePlainsConfig(SceneConfig):
    cluster_period: int = 11  # distance between cluster anchors
    cluster_radius: int = 1  # manhattan radius of a cluster
    cluster_prob: float = 0.85  # chance to place a cluster at an anchor
    jitter: int = 2  # random offset applied per anchor


class BiomePlains(Scene[BiomePlainsConfig]):
    def render(self) -> None:
        grid = self.grid
        H, W = self.height, self.width
        p = self.config

        period = max(3, int(p.cluster_period))
        radius = max(0, int(p.cluster_radius))
        jitter = max(0, int(p.jitter))

        rocks = np.zeros((H, W), dtype=bool)

        for y0 in range(0, H, period):
            for x0 in range(0, W, period):
                if self.rng.random() > float(p.cluster_prob):
                    continue

                dx = int(self.rng.integers(-jitter, jitter + 1)) if jitter > 0 else 0
                dy = int(self.rng.integers(-jitter, jitter + 1)) if jitter > 0 else 0
                cx = x0 + dx
                cy = y0 + dy
                if cx < 0 or cy < 0 or cx >= W or cy >= H:
                    continue

                for oy in range(-radius, radius + 1):
                    for ox in range(-radius, radius + 1):
                        if abs(ox) + abs(oy) > radius:
                            continue
                        x = cx + ox
                        y = cy + oy
                        if 0 <= x < W and 0 <= y < H:
                            rocks[y, x] = True

        grid[rocks] = "wall"
