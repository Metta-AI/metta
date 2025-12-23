import numpy as np

from mettagrid.mapgen.scene import Scene, SceneConfig


class BiomePlainsConfig(SceneConfig):
    cluster_period: int = 7  # distance between cluster anchors
    cluster_min_radius: int = 1  # min radius for a cluster
    cluster_max_radius: int = 2  # max radius for a cluster
    cluster_fill: float = 0.7  # density of rocks within a cluster footprint
    cluster_prob: float = 0.8  # chance to place a cluster at an anchor
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

        directions = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)], dtype=int)

        for cx, cy in anchors:
            radius = int(self.rng.integers(min_radius, max_radius + 1)) if max_radius > 0 else 0
            if radius == 0:
                rocks[cy, cx] = True
                continue

            fill = float(p.cluster_fill) * float(self.rng.uniform(0.6, 1.0))
            branch_count = int(self.rng.integers(2, 5))
            max_steps = max(3, radius * 3)
            max_dist2 = (radius + 1) * (radius + 1)

            x = np.full(branch_count, cx, dtype=int)
            y = np.full(branch_count, cy, dtype=int)
            dir_idx = self.rng.integers(0, 4, size=branch_count)

            for step in range(max_steps):
                in_bounds = (x >= 0) & (x < W) & (y >= 0) & (y < H)
                place = in_bounds & (self.rng.random(branch_count) <= fill)
                if place.any():
                    rocks[y[place], x[place]] = True

                turn = self.rng.random(branch_count) < 0.35
                if turn.any():
                    dir_idx[turn] = self.rng.integers(0, 4, size=int(turn.sum()))

                dx = directions[dir_idx, 0]
                dy = directions[dir_idx, 1]
                nx = x + dx
                ny = y + dy

                out = (nx - cx) * (nx - cx) + (ny - cy) * (ny - cy) > max_dist2
                if out.any():
                    dir_idx[out] = self.rng.integers(0, 4, size=int(out.sum()))
                    dx = directions[dir_idx, 0]
                    dy = directions[dir_idx, 1]
                    nx = x + dx
                    ny = y + dy

                x, y = nx, ny

                if step > 1:
                    spur = self.rng.random(branch_count) < 0.12
                    if spur.any():
                        spur_dirs = self.rng.integers(0, 4, size=int(spur.sum()))
                        sdx = directions[spur_dirs, 0]
                        sdy = directions[spur_dirs, 1]
                        sx = x[spur] + sdx
                        sy = y[spur] + sdy
                        dist2 = (sx - cx) * (sx - cx) + (sy - cy) * (sy - cy)
                        spur_ok = dist2 <= max_dist2
                        if spur_ok.any():
                            spur_in = (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
                            spur_place = spur_ok & spur_in & (self.rng.random(len(sx)) <= fill)
                            if spur_place.any():
                                rocks[sy[spur_place], sx[spur_place]] = True

                            sx2 = sx + sdx
                            sy2 = sy + sdy
                            dist2_2 = (sx2 - cx) * (sx2 - cx) + (sy2 - cy) * (sy2 - cy)
                            spur_ok_2 = spur_ok & (dist2_2 <= max_dist2)
                            if spur_ok_2.any():
                                spur_in_2 = (sx2 >= 0) & (sx2 < W) & (sy2 >= 0) & (sy2 < H)
                                spur_place_2 = spur_ok_2 & spur_in_2 & (self.rng.random(len(sx2)) <= fill)
                                if spur_place_2.any():
                                    rocks[sy2[spur_place_2], sx2[spur_place_2]] = True

        grid[rocks] = "wall"
