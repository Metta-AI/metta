import numpy as np

from mettagrid.mapgen.scene import Scene, SceneConfig


class BiomePlainsConfig(SceneConfig):
    cluster_period: int = 9  # distance between cluster anchors
    cluster_min_radius: int = 1  # min radius for a cluster
    cluster_max_radius: int = 3  # max radius for a cluster
    cluster_fill: float = 0.6  # density of rocks within a cluster footprint
    cluster_prob: float = 0.7  # chance to place a cluster at an anchor
    jitter: int = 2  # random offset applied per anchor


class BiomePlains(Scene[BiomePlainsConfig]):
    def render(self) -> None:
        grid = self.grid
        H, W = self.height, self.width
        p = self.config

        period = max(3, int(p.cluster_period))
        min_radius = max(0, int(p.cluster_min_radius))
        max_radius = max(min_radius, int(p.cluster_max_radius))
        jitter = max(0, int(p.jitter))

        rocks = np.zeros((H, W), dtype=bool)

        ys = np.arange(0, H, period, dtype=int)
        xs = np.arange(0, W, period, dtype=int)
        if ys.size == 0 or xs.size == 0:
            return

        grid_x, grid_y = np.meshgrid(xs, ys)
        anchors = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
        keep = self.rng.random(len(anchors)) <= float(p.cluster_prob)
        anchors = anchors[keep]
        if anchors.size == 0:
            return

        if jitter > 0:
            anchors = anchors + self.rng.integers(-jitter, jitter + 1, size=anchors.shape)

        in_bounds = (anchors[:, 0] >= 0) & (anchors[:, 0] < W) & (anchors[:, 1] >= 0) & (anchors[:, 1] < H)
        anchors = anchors[in_bounds]
        if anchors.size == 0:
            return

        for cx, cy in anchors:
            radius = int(self.rng.integers(min_radius, max_radius + 1)) if max_radius > 0 else 0
            if radius == 0:
                rocks[cy, cx] = True
                continue

            fill = float(p.cluster_fill) * float(self.rng.uniform(0.6, 1.0))
            branch_count = int(self.rng.integers(2, 5))
            max_steps = max(3, radius * 3)
            max_dist2 = (radius + 1) * (radius + 1)

            for _ in range(branch_count):
                x, y = cx, cy
                dx, dy = self.rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
                for step in range(max_steps):
                    if 0 <= x < W and 0 <= y < H and self.rng.random() <= fill:
                        rocks[y, x] = True

                    if self.rng.random() < 0.35:
                        dx, dy = self.rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])

                    nx = x + dx
                    ny = y + dy
                    if (nx - cx) * (nx - cx) + (ny - cy) * (ny - cy) > max_dist2:
                        dx, dy = self.rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
                        nx = x + dx
                        ny = y + dy

                    x, y = nx, ny

                    if self.rng.random() < 0.12 and step > 1:
                        sx, sy = x, y
                        sdx, sdy = self.rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
                        for _ in range(2):
                            sx += sdx
                            sy += sdy
                            if (sx - cx) * (sx - cx) + (sy - cy) * (sy - cy) > max_dist2:
                                break
                            if 0 <= sx < W and 0 <= sy < H and self.rng.random() <= fill:
                                rocks[sy, sx] = True

        grid[rocks] = "wall"
