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

        def _cut_triangle_from_top(x: int) -> None:
            if self.rng.random() >= chunk_prob:
                return
            depth = int(self.rng.integers(depth_min, depth_max + 1))
            half_w = int(self.rng.integers(width_min, width_max + 1))
            if depth <= 0 or half_w <= 0:
                return
            for dy in range(depth):
                span = max(0, int(round(half_w * (1.0 - dy / max(1.0, depth)))))
                if span == 0:
                    continue
                x0 = max(0, x - span)
                x1 = min(width, x + span + 1)
                grid[dy, x0:x1] = "wall"

        def _cut_triangle_from_bottom(x: int) -> None:
            if self.rng.random() >= chunk_prob:
                return
            depth = int(self.rng.integers(depth_min, depth_max + 1))
            half_w = int(self.rng.integers(width_min, width_max + 1))
            if depth <= 0 or half_w <= 0:
                return
            for dy in range(depth):
                span = max(0, int(round(half_w * (1.0 - dy / max(1.0, depth)))))
                if span == 0:
                    continue
                x0 = max(0, x - span)
                x1 = min(width, x + span + 1)
                y = height - 1 - dy
                grid[y, x0:x1] = "wall"

        def _cut_triangle_from_left(y: int) -> None:
            if self.rng.random() >= chunk_prob:
                return
            depth = int(self.rng.integers(depth_min, depth_max + 1))
            half_w = int(self.rng.integers(width_min, width_max + 1))
            if depth <= 0 or half_w <= 0:
                return
            for dx in range(depth):
                span = max(0, int(round(half_w * (1.0 - dx / max(1.0, depth)))))
                if span == 0:
                    continue
                y0 = max(0, y - span)
                y1 = min(height, y + span + 1)
                grid[y0:y1, dx] = "wall"

        def _cut_triangle_from_right(y: int) -> None:
            if self.rng.random() >= chunk_prob:
                return
            depth = int(self.rng.integers(depth_min, depth_max + 1))
            half_w = int(self.rng.integers(width_min, width_max + 1))
            if depth <= 0 or half_w <= 0:
                return
            for dx in range(depth):
                span = max(0, int(round(half_w * (1.0 - dx / max(1.0, depth)))))
                if span == 0:
                    continue
                y0 = max(0, y - span)
                y1 = min(height, y + span + 1)
                x = width - 1 - dx
                grid[y0:y1, x] = "wall"

        for x in range(0, width, step):
            _cut_triangle_from_top(x)
            _cut_triangle_from_bottom(x)

        for y in range(0, height, step):
            _cut_triangle_from_left(y)
            _cut_triangle_from_right(y)
