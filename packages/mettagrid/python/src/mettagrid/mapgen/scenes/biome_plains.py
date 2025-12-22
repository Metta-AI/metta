import numpy as np

from mettagrid.mapgen.scene import Scene, SceneConfig


class BiomePlainsConfig(SceneConfig):
    cluster_period: int = 7  # distance between cluster anchors
    cluster_min_radius: int = 1  # min radius for a cluster
    cluster_max_radius: int = 3  # max radius for a cluster
    cluster_fill: float = 0.7  # density of rocks within a cluster footprint
    cluster_prob: float = 0.85  # chance to place a cluster at an anchor
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

        offsets_cache: dict[int, np.ndarray] = {}
        for radius in range(min_radius, max_radius + 1):
            if radius == 0:
                offsets_cache[radius] = np.array([[0, 0]], dtype=int)
                continue
            dy, dx = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1), indexing="ij")
            mask = (dx * dx + dy * dy) <= radius * radius
            offsets_cache[radius] = np.stack([dx[mask], dy[mask]], axis=1).astype(int)

        for cx, cy in anchors:
            radius = int(self.rng.integers(min_radius, max_radius + 1)) if max_radius > 0 else 0
            offsets = offsets_cache[radius]
            if radius > 0:
                fill = float(p.cluster_fill) * float(self.rng.uniform(0.6, 1.0))
                if fill < 1.0:
                    offsets = offsets[self.rng.random(len(offsets)) <= fill]
                    if offsets.size == 0:
                        continue
            coords = offsets + np.array([cx, cy])
            mask = (coords[:, 0] >= 0) & (coords[:, 0] < W) & (coords[:, 1] >= 0) & (coords[:, 1] < H)
            coords = coords[mask]
            if coords.size == 0:
                continue
            rocks[coords[:, 1], coords[:, 0]] = True

        grid[rocks] = "wall"
