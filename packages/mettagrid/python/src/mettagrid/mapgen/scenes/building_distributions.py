from __future__ import annotations

from enum import Enum

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_serializer

from mettagrid.mapgen.scene import Scene, SceneConfig

DEFAULT_BUILDING_WEIGHTS: dict[str, float] = {
    "charger": 0.3,
    "silicon_extractor": 0.2,
    "carbon_extractor": 0.1,
    "oxygen_extractor": 0.1,
    "germanium_extractor": 0.1,
}
DEFAULT_FALLBACK_WEIGHT = 0.1


class DistributionType(str, Enum):
    """Types of spatial distributions for building placement."""

    UNIFORM = "uniform"
    NORMAL = "normal"
    EXPONENTIAL = "exponential"
    POISSON = "poisson"
    BIMODAL = "bimodal"


class DistributionConfig(BaseModel):
    """Configuration for spatial distribution of buildings."""

    type: DistributionType = DistributionType.UNIFORM
    # Normal/Gaussian parameters
    mean_x: float | None = None  # Center x (fraction 0-1, None = center)
    mean_y: float | None = None  # Center y (fraction 0-1, None = center)
    std_x: float = 0.2  # Standard deviation x (fraction of width)
    std_y: float = 0.2  # Standard deviation y (fraction of height)
    # Exponential parameters
    decay_rate: float = 2.0  # How quickly density falls off
    origin_x: float = 0.0  # Starting edge (0=left, 1=right)
    origin_y: float = 0.0  # Starting edge (0=top, 1=bottom)
    # Bimodal parameters
    center1_x: float = 0.25  # First cluster center x
    center1_y: float = 0.25  # First cluster center y
    center2_x: float = 0.75  # Second cluster center x
    center2_y: float = 0.75  # Second cluster center y
    cluster_std: float = 0.15  # Standard deviation for each cluster

    @field_serializer("type")
    def _ser_type(self, value: DistributionType) -> str:
        return value.value


def _sample_positions_by_distribution(
    count: int,
    width: int,
    height: int,
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
    dist_config: DistributionConfig,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Sample positions according to the specified distribution.

    Returns list of (row, col) tuples within the specified bounds.
    """
    if count <= 0 or width <= 0 or height <= 0:
        return []

    available_width = col_max - col_min + 1
    available_height = row_max - row_min + 1

    if available_width <= 0 or available_height <= 0:
        return []

    dist_type = dist_config.type

    if dist_type == DistributionType.UNIFORM:
        # Original uniform grid behavior
        rows = rng.integers(row_min, row_max + 1, size=count)
        cols = rng.integers(col_min, col_max + 1, size=count)
        return list(zip(rows.tolist(), cols.tolist(), strict=False))

    elif dist_type == DistributionType.NORMAL:
        # Gaussian distribution centered on mean
        mean_x = dist_config.mean_x if dist_config.mean_x is not None else 0.5
        mean_y = dist_config.mean_y if dist_config.mean_y is not None else 0.5

        center_col = col_min + mean_x * available_width
        center_row = row_min + mean_y * available_height
        std_col = dist_config.std_x * available_width
        std_row = dist_config.std_y * available_height

        cols = rng.normal(center_col, std_col, size=count)
        rows = rng.normal(center_row, std_row, size=count)

        cols = np.clip(cols, col_min, col_max).astype(int)
        rows = np.clip(rows, row_min, row_max).astype(int)

        return list(zip(rows.tolist(), cols.tolist(), strict=False))

    elif dist_type == DistributionType.EXPONENTIAL:
        # Exponential decay from origin
        samples_x = rng.exponential(scale=1.0 / dist_config.decay_rate, size=count)
        samples_y = rng.exponential(scale=1.0 / dist_config.decay_rate, size=count)

        # Normalize to [0, 1]
        samples_x = np.clip(samples_x, 0, 1)
        samples_y = np.clip(samples_y, 0, 1)

        # Apply origin offset and flip if needed
        if dist_config.origin_x > 0.5:
            samples_x = 1.0 - samples_x
        if dist_config.origin_y > 0.5:
            samples_y = 1.0 - samples_y

        cols = (col_min + samples_x * available_width).astype(int)
        rows = (row_min + samples_y * available_height).astype(int)

        cols = np.clip(cols, col_min, col_max)
        rows = np.clip(rows, row_min, row_max)

        return list(zip(rows.tolist(), cols.tolist(), strict=False))

    elif dist_type == DistributionType.POISSON:
        # Poisson process - uniform but with natural clumping
        # Use small clusters randomly distributed
        num_clusters = max(1, count // 5)
        cluster_centers_x = rng.uniform(col_min, col_max, size=num_clusters)
        cluster_centers_y = rng.uniform(row_min, row_max, size=num_clusters)

        positions = []
        for _ in range(count):
            cluster_idx = rng.integers(0, num_clusters)
            center_x = cluster_centers_x[cluster_idx]
            center_y = cluster_centers_y[cluster_idx]

            # Add small jitter around cluster center
            jitter_x = rng.normal(0, available_width * 0.05)
            jitter_y = rng.normal(0, available_height * 0.05)

            col = int(np.clip(center_x + jitter_x, col_min, col_max))
            row = int(np.clip(center_y + jitter_y, row_min, row_max))
            positions.append((row, col))

        return positions

    elif dist_type == DistributionType.BIMODAL:
        # Two Gaussian clusters
        half = count // 2
        remainder = count - half

        center1_col = col_min + dist_config.center1_x * available_width
        center1_row = row_min + dist_config.center1_y * available_height
        center2_col = col_min + dist_config.center2_x * available_width
        center2_row = row_min + dist_config.center2_y * available_height

        std_col = dist_config.cluster_std * available_width
        std_row = dist_config.cluster_std * available_height

        cols1 = rng.normal(center1_col, std_col, size=half)
        rows1 = rng.normal(center1_row, std_row, size=half)

        cols2 = rng.normal(center2_col, std_col, size=remainder)
        rows2 = rng.normal(center2_row, std_row, size=remainder)

        cols = np.concatenate([cols1, cols2])
        rows = np.concatenate([rows1, rows2])

        cols = np.clip(cols, col_min, col_max).astype(int)
        rows = np.clip(rows, row_min, row_max).astype(int)

        return list(zip(rows.tolist(), cols.tolist(), strict=False))

    # Fallback to uniform
    rows = rng.integers(row_min, row_max + 1, size=count)
    cols = rng.integers(col_min, col_max + 1, size=count)
    return list(zip(rows.tolist(), cols.tolist(), strict=False))


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
    building_names: list[str] = Field(
        default_factory=lambda: [
            "carbon_extractor",
            "oxygen_extractor",
            "germanium_extractor",
            "silicon_extractor",
            "charger",
        ]
    )
    building_weights: dict[str, float] | None = None
    # Spatial distribution configuration
    distribution: DistributionConfig = Field(default_factory=lambda: DistributionConfig())
    # Per-building-type distribution overrides
    building_distributions: dict[str, DistributionConfig] | None = None


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

        names, probabilities = self._resolve_building_distribution()

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

            placed_centers_tc: list[tuple[int, int]] = []

            # Check if per-building-type distributions are specified
            if params.building_distributions:
                # Group buildings by their distribution
                building_groups: dict[str, list[str]] = {}
                for name in names:
                    if name in params.building_distributions:
                        # Use per-building distribution
                        dist_key = str(params.building_distributions[name].model_dump())
                        if dist_key not in building_groups:
                            building_groups[dist_key] = []
                        building_groups[dist_key].append(name)
                    else:
                        # Use default distribution
                        default_key = str(params.distribution.model_dump())
                        if default_key not in building_groups:
                            building_groups[default_key] = []
                        building_groups[default_key].append(name)

                # Calculate how many buildings of each type to place
                total_weight = sum(probabilities)
                for _dist_key, group_names in building_groups.items():
                    # Get the distribution config for this group
                    first_name = group_names[0]
                    dist_config = params.building_distributions.get(first_name, params.distribution)

                    # Calculate weight for this group
                    group_indices = [i for i, name in enumerate(names) if name in group_names]
                    group_weight = sum(probabilities[i] for i in group_indices)
                    group_count = max(1, int((group_weight / total_weight) * placement_goal))

                    # Sample positions for this group
                    group_positions = _sample_positions_by_distribution(
                        count=group_count,
                        width=self.width,
                        height=self.height,
                        row_min=row_min,
                        row_max=row_max,
                        col_min=col_min,
                        col_max=col_max,
                        dist_config=dist_config,
                        rng=self.rng,
                    )

                    # Create probabilities for just this group
                    group_probs = np.array([probabilities[i] for i in group_indices])
                    group_probs = group_probs / np.sum(group_probs)

                    # Assign building types within this group
                    group_assignments = self._sample_assignments(len(group_positions), group_names, group_probs)

                    # Place buildings
                    for (row, col), name in zip(group_positions, group_assignments, strict=False):
                        if not can_place(row, col, placed_centers_tc):
                            continue
                        if carve_and_place(row, col, name):
                            placed_centers_tc.append((row, col))
            else:
                # Use single distribution for all buildings
                dist_config = params.distribution
                positions = _sample_positions_by_distribution(
                    count=placement_goal,
                    width=self.width,
                    height=self.height,
                    row_min=row_min,
                    row_max=row_max,
                    col_min=col_min,
                    col_max=col_max,
                    dist_config=dist_config,
                    rng=self.rng,
                )

                if not positions:
                    return

                assignments = self._sample_assignments(len(positions), names, probabilities)

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

    def _resolve_building_distribution(self) -> tuple[list[str], NDArray[np.float64]]:
        weights = self.config.building_weights
        if weights:
            filtered = [(name, float(weight)) for name, weight in weights.items() if float(weight) > 0]
            if not filtered:
                raise ValueError("building_weights must contain positive values")
            names, raw_weights = zip(*filtered, strict=False)
            weight_array = np.asarray(raw_weights, dtype=float)
        else:
            names = self.config.building_names or ["carbon_extractor"]
            if not names:
                raise ValueError("At least one extractor name must be provided")
            weight_array = np.asarray(
                [DEFAULT_BUILDING_WEIGHTS.get(name, DEFAULT_FALLBACK_WEIGHT) for name in names],
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
