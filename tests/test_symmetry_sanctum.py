import numpy as np

from cogames.cogs_vs_clips.scenarios import machina_sanctum, machina_symmetry_sanctum

CONVERTER_TYPES = {"generator_red", "generator_blue", "generator_green", "lab"}


def build_grid(cfg_maker, seed: int):
    cfg = cfg_maker(num_cogs=4)
    cfg.game.map_builder.seed = seed
    mb = cfg.game.map_builder.create()
    level = mb.build()
    return level.grid


def find_altar_center(grid: np.ndarray) -> tuple[int, int]:
    ys, xs = np.where(grid == "altar")
    assert ys.size > 0, "altar not found"
    return int(ys[0]), int(xs[0])


def counts_by_type(grid: np.ndarray) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in CONVERTER_TYPES:
        counts[t] = int(np.count_nonzero(grid == t))
    return counts


def mask_out_base(grid: np.ndarray, base_half_extent: int = 5) -> np.ndarray:
    h, w = grid.shape
    cy, cx = find_altar_center(grid)
    y0 = max(0, cy - base_half_extent)
    y1 = min(h, cy + base_half_extent + 1)
    x0 = max(0, cx - base_half_extent)
    x1 = min(w, cx + base_half_extent + 1)
    mask = np.ones_like(grid, dtype=bool)
    mask[y0:y1, x0:x1] = False
    return mask


def is_converter(name: str) -> bool:
    return name in CONVERTER_TYPES


def test_machina_sanctum_converter_counts_detached_from_base():
    grid = build_grid(machina_sanctum, seed=12345)
    # Exclude 11x11 base region around altar to validate quadrant placement counts
    mask = mask_out_base(grid, base_half_extent=5)
    masked = np.where(mask, grid, "empty")
    counts = counts_by_type(masked)
    # Expect at least 3 per type outside base (scenario target)
    for t in CONVERTER_TYPES:
        assert counts[t] >= 3, f"Expected at least 3 of {t} outside base, got {counts[t]}"


def test_machina_symmetry_sanctum_symmetry_and_counts():
    grid = build_grid(machina_symmetry_sanctum, seed=67890)
    h, w = grid.shape
    # Symmetry check: both axes — if a converter at (y,x), then converters at mirror positions
    for y in range(h):
        for x in range(w):
            if is_converter(grid[y, x]):
                if w > 1:
                    mx = w - 1 - x
                    assert is_converter(grid[y, mx]), "Horizontal symmetry violated for converter placement"
                if h > 1:
                    my = h - 1 - y
                    assert is_converter(grid[my, x]), "Vertical symmetry violated for converter placement"

    # Counts per type should meet target totals (tolerate >= in case base adds fixed converters)
    counts = counts_by_type(grid)
    for t in CONVERTER_TYPES:
        assert counts[t] >= 3, f"Expected at least 3 of {t} overall, got {counts[t]}"
