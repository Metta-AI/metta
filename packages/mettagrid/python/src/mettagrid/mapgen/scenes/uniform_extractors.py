from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from mettagrid.mapgen.scene import Scene, SceneConfig

DEFAULT_EXTRACTOR_WEIGHTS: dict[str, float] = {
    "charger": 0.3,
    "silicon_extractor": 0.2,
    "carbon_extractor": 0.1,
    "oxygen_extractor": 0.1,
    "germanium_extractor": 0.1,
}
DEFAULT_FALLBACK_WEIGHT = 0.1


def _linspace_positions(count: int, interior_size: int) -> list[int]:
    """Return approximately uniform interior positions for the given count."""
    if count <= 0:
        return []
    if interior_size <= 0:
        raise ValueError("interior_size must be positive")

    if count >= interior_size:
        return [i for i in range(1, interior_size + 1)]

    step = (interior_size + 1) / (count + 1)
    return [1 + max(0, min(interior_size - 1, round(step * (i + 1)))) for i in range(count)]


class UniformExtractorParams(SceneConfig):
    rows: int = 4
    cols: int = 4
    jitter: int = 1
    padding: int = 1
    clear_existing: bool = False
    frame_with_walls: bool = False
    target_coverage: float | None = None
    extractor_names: list[str] = Field(
        default_factory=lambda: [
            "carbon_extractor",
            "oxygen_extractor",
            "germanium_extractor",
            "silicon_extractor",
            "charger",
        ]
    )
    extractor_weights: dict[str, float] | None = None


class UniformExtractorScene(Scene[UniformExtractorParams]):
    """Place extractor stations on a jittered uniform grid."""

    def render(self) -> None:
        params = self.config
        if self.width < 3 or self.height < 3:
            raise ValueError("Extractor map must be at least 3x3 to fit border walls")

        padding = max(0, params.padding)
        row_min = padding
        row_max = self.height - padding - 1
        col_min = padding
        col_max = self.width - padding - 1

        if row_min > row_max or col_min > col_max:
            return

        if params.clear_existing:
            self.grid[:, :] = "empty"
            if params.frame_with_walls:
                self.grid[0, :] = "wall"
                self.grid[-1, :] = "wall"
                self.grid[:, 0] = "wall"
                self.grid[:, -1] = "wall"

        interior_width = self.width - 2
        interior_height = self.height - 2

        spacing = padding + 1

        grid = self.grid

        names, probabilities = self._resolve_extractor_distribution()

        def carve_and_place(center_row: int, center_col: int, name: str) -> bool:
            for rr in range(center_row - padding, center_row + padding + 1):
                if rr < 0 or rr >= self.height:
                    continue
                for cc in range(center_col - padding, center_col + padding + 1):
                    if cc < 0 or cc >= self.width:
                        continue
                    if rr == center_row and cc == center_col:
                        grid[rr, cc] = name
                    else:
                        grid[rr, cc] = "empty"

            return True

        def can_place(center_row: int, center_col: int, centers: list[tuple[int, int]]) -> bool:
            return not any(abs(center_row - r0) <= padding and abs(center_col - c0) <= padding for r0, c0 in centers)

        if params.target_coverage is not None:
            available_height = row_max - row_min + 1
            available_width = col_max - col_min + 1
            if available_height <= 0 or available_width <= 0:
                return

            max_rows = max(0, (available_height + spacing - 1) // spacing)
            max_cols = max(0, (available_width + spacing - 1) // spacing)
            max_possible = max_rows * max_cols
            if max_possible == 0:
                return

            desired = int(params.target_coverage * interior_width * interior_height)
            placement_goal = min(max_possible, max(1, desired))

            valid_row_starts = [row_min + offset for offset in range(spacing) if row_min + offset <= row_max]
            valid_col_starts = [col_min + offset for offset in range(spacing) if col_min + offset <= col_max]
            if not valid_row_starts or not valid_col_starts:
                return

            start_row = int(self.rng.choice(valid_row_starts))
            start_col = int(self.rng.choice(valid_col_starts))

            rows = list(range(start_row, row_max + 1, spacing))
            cols = list(range(start_col, col_max + 1, spacing))
            positions = [(r, c) for r in rows for c in cols]
            if not positions:
                return

            positions = positions[:max_possible]
            permutation = self.rng.permutation(len(positions))
            positions = [positions[i] for i in permutation]
            positions = positions[:placement_goal]

            #sample assignments draws extractor types, which are numpy.random.choice with the normalized weights from each distribution type
            assignments = self._sample_assignments(len(positions), names, probabilities)

            placed_centers_tc: list[tuple[int, int]] = []
            for (row, col), name in zip(positions, assignments, strict=False):
                if not can_place(row, col, placed_centers_tc):
                    continue
                if carve_and_place(row, col, name):
                    placed_centers_tc.append((row, col))
            return

        row_positions = _linspace_positions(params.rows, interior_height)
        col_positions = _linspace_positions(params.cols, interior_width)

        if not row_positions or not col_positions:
            raise ValueError("rows and cols must be positive for extractor placement")

        raw_positions = [(row, col) for row in row_positions for col in col_positions]
        positions: list[tuple[int, int]] = []
        seen = set()
        for row, col in raw_positions:
            if (row, col) not in seen:
                seen.add((row, col))
                positions.append((row, col))

        if not positions:
            return

        assignments = self._sample_assignments(len(positions), names, probabilities)

        jitter = max(0, params.jitter)
        placed_centers: list[tuple[int, int]] = []
        for (base_row, base_col), name in zip(positions, assignments, strict=False):
            row = int(min(row_max, max(row_min, base_row)))
            col = int(min(col_max, max(col_min, base_col)))
            attempts = max(1, 8 if jitter else 1)
            placement: tuple[int, int] | None = None
            for _ in range(attempts):
                offset_row = int(
                    np.clip(
                        row + (self.rng.integers(-jitter, jitter + 1) if jitter else 0),
                        row_min,
                        row_max,
                    )
                )
                offset_col = int(
                    np.clip(
                        col + (self.rng.integers(-jitter, jitter + 1) if jitter else 0),
                        col_min,
                        col_max,
                    )
                )
                if not (row_min <= offset_row <= row_max and col_min <= offset_col <= col_max):
                    continue
                if not can_place(offset_row, offset_col, placed_centers):
                    continue
                placement = (offset_row, offset_col)
                break
            if placement is None:
                continue
            row, col = placement
            if carve_and_place(row, col, name):
                placed_centers.append((row, col))

    def _resolve_extractor_distribution(self) -> tuple[list[str], NDArray[np.float64]]:
        weights = self.config.extractor_weights
        if weights:
            filtered = [(name, float(weight)) for name, weight in weights.items() if float(weight) > 0]
            if not filtered:
                raise ValueError("extractor_weights must contain positive values")
            names, raw_weights = zip(*filtered, strict=False)
            weight_array = np.asarray(raw_weights, dtype=float)
        else:
            names = self.config.extractor_names or ["carbon_extractor"]
            if not names:
                raise ValueError("At least one extractor name must be provided")
            weight_array = np.asarray(
                [DEFAULT_EXTRACTOR_WEIGHTS.get(name, DEFAULT_FALLBACK_WEIGHT) for name in names],
                dtype=float,
            )

        total = float(np.sum(weight_array))
        if total <= 0:
            raise ValueError("Sum of extractor weights must be positive")

        probabilities = weight_array / total
        return list(names), probabilities.astype(float)

    def _sample_assignments(self, count: int, names: list[str], probabilities: NDArray[np.float64]) -> list[str]:
        if count <= 0:
            return []
        return list(self.rng.choice(names, size=count, replace=True, p=probabilities))
