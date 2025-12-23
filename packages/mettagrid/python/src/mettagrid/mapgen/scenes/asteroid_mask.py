from mettagrid.mapgen.scene import Scene, SceneConfig


class AsteroidMaskConfig(SceneConfig):
    step: int = 3
    depth_min: int = 2
    depth_max: int = 8
    width_min: int = 2
    width_max: int = 6
    chunk_prob: float = 0.6


class AsteroidMask(Scene[AsteroidMaskConfig]):
    def render(self) -> None:
        grid = self.grid
        height, width = self.height, self.width
        cfg = self.config

        step = max(1, int(cfg.step))
        depth_min = max(0, int(cfg.depth_min))
        depth_max = max(depth_min, int(cfg.depth_max))
        width_min = max(0, int(cfg.width_min))
        width_max = max(width_min, int(cfg.width_max))
        chunk_prob = float(cfg.chunk_prob)

        if depth_max == 0 or width_max == 0 or chunk_prob <= 0.0:
            return

        def _cut_triangle(anchor: int, depth: int, half_w: int, axis: str, reverse: bool) -> None:
            if depth <= 0 or half_w <= 0:
                return
            denom = max(1, depth)
            for offset in range(depth):
                span = int(round(half_w * (1.0 - offset / denom)))
                if span <= 0:
                    continue
                if axis == "x":
                    x0 = max(0, anchor - span)
                    x1 = min(width, anchor + span + 1)
                    y = height - 1 - offset if reverse else offset
                    grid[y, x0:x1] = "wall"
                else:
                    y0 = max(0, anchor - span)
                    y1 = min(height, anchor + span + 1)
                    x = width - 1 - offset if reverse else offset
                    grid[y0:y1, x] = "wall"

        for x in range(0, width, step):
            if self.rng.random() < chunk_prob:
                _cut_triangle(
                    x,
                    int(self.rng.integers(depth_min, depth_max + 1)),
                    int(self.rng.integers(width_min, width_max + 1)),
                    "x",
                    False,
                )
            if self.rng.random() < chunk_prob:
                _cut_triangle(
                    x,
                    int(self.rng.integers(depth_min, depth_max + 1)),
                    int(self.rng.integers(width_min, width_max + 1)),
                    "x",
                    True,
                )

        for y in range(0, height, step):
            if self.rng.random() < chunk_prob:
                _cut_triangle(
                    y,
                    int(self.rng.integers(depth_min, depth_max + 1)),
                    int(self.rng.integers(width_min, width_max + 1)),
                    "y",
                    False,
                )
            if self.rng.random() < chunk_prob:
                _cut_triangle(
                    y,
                    int(self.rng.integers(depth_min, depth_max + 1)),
                    int(self.rng.integers(width_min, width_max + 1)),
                    "y",
                    True,
                )
