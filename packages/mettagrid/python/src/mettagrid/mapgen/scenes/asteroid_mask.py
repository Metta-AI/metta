import numpy as np

from mettagrid.mapgen.scene import Scene, SceneConfig


class AsteroidMaskConfig(SceneConfig):
    inset_min: int = 2
    inset_max: int = 12
    jagged_step: int = 1
    spike_prob: float = 0.25
    spike_depth: int = 8
    smooth: int = 3


class AsteroidMask(Scene[AsteroidMaskConfig]):
    def render(self) -> None:
        grid = self.grid
        height, width = self.height, self.width
        cfg = self.config

        min_inset = max(0, int(cfg.inset_min))
        max_inset = max(min_inset, int(cfg.inset_max))
        jagged_step = max(0, int(cfg.jagged_step))
        spike_prob = float(cfg.spike_prob)
        spike_depth = max(0, int(cfg.spike_depth))
        smooth = max(1, int(cfg.smooth))

        def _make_depths(length: int, cap: int) -> np.ndarray:
            if cap <= 0 or length <= 0:
                return np.zeros(length, dtype=np.int32)
            cur = int(self.rng.integers(min_inset, cap + 1)) if cap > min_inset else min_inset
            depths = np.empty(length, dtype=np.int32)
            for i in range(length):
                if spike_depth > 0 and self.rng.random() < spike_prob:
                    cur = min(cap, cur + int(self.rng.integers(1, spike_depth + 1)))
                elif jagged_step > 0:
                    cur += int(self.rng.integers(-jagged_step, jagged_step + 1))
                if self.rng.random() < 0.1:
                    cur += int(self.rng.integers(-2, 3))
                cur = int(np.clip(cur, min_inset, cap))
                depths[i] = cur
            if smooth > 1 and length >= smooth:
                kernel = np.ones(smooth, dtype=np.float32) / float(smooth)
                padded = np.pad(depths.astype(np.float32), (smooth, smooth), mode="edge")
                smoothed = np.convolve(padded, kernel, mode="same")[smooth:-smooth]
                depths = np.clip(smoothed, min_inset, cap).astype(np.int32)
            return depths

        max_top = min(max_inset, height // 2)
        max_side = min(max_inset, width // 2)

        top = _make_depths(width, max_top)
        bottom = _make_depths(width, max_top)
        left = _make_depths(height, max_side)
        right = _make_depths(height, max_side)

        for x, depth in enumerate(top):
            if depth > 0:
                grid[:depth, x] = "wall"
        for x, depth in enumerate(bottom):
            if depth > 0:
                grid[height - depth :, x] = "wall"
        for y, depth in enumerate(left):
            if depth > 0:
                grid[y, :depth] = "wall"
        for y, depth in enumerate(right):
            if depth > 0:
                grid[y, width - depth :] = "wall"
