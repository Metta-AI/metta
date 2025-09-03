"""ASCII map transformation utilities for rotation, mirroring, and stretching.

This module provides functions to transform ASCII maps while preserving
all character semantics including converter chains, agents, and special objects.

Available transformations:
- Rotation: 90°, 180°, 270° clockwise
- Mirroring: horizontal (left-right) and vertical (top-bottom)
- Stretching: scale maps by integer factors (2x, 3x, etc.)
  * Only walls and empty spaces are duplicated
  * Special objects (agents, converters) appear once in their stretched cell
"""

from dataclasses import dataclass
from typing import Callable, Iterable, List, Literal, Optional

import numpy as np

from metta.mettagrid.mapgen.utils.ascii_grid import char_grid_to_lines


@dataclass(frozen=True)
class Transform:
    """Describes a single map transformation and its naming suffix."""

    name: str
    suffix: str
    apply: Callable[[str], str]  # takes map content, returns transformed content


# Families of single transformations (clockwise rotations, flips, and 2x stretch)
# Comment out transformations you don't want to use
ROTATIONS: list[Transform] = [
    Transform(name="rotate", suffix="90", apply=lambda s: rotate_ascii_map(s, 90)),
    Transform(name="rotate", suffix="180", apply=lambda s: rotate_ascii_map(s, 180)),
    Transform(name="rotate", suffix="270", apply=lambda s: rotate_ascii_map(s, 270)),
]

MIRRORS: list[Transform] = [
    Transform(name="mirror", suffix="hflip", apply=lambda s: mirror_ascii_map(s, "horizontal")),
    Transform(name="mirror", suffix="vflip", apply=lambda s: mirror_ascii_map(s, "vertical")),
]

STRETCHES: list[Transform] = [
    Transform(
        name="stretch",
        suffix="sx2",
        apply=lambda s: stretch_ascii_map(s, scale_x=2, scale_y=1),
    ),
    Transform(
        name="stretch",
        suffix="sy2",
        apply=lambda s: stretch_ascii_map(s, scale_x=1, scale_y=2),
    ),
    Transform(
        name="stretch",
        suffix="sxy2",
        apply=lambda s: stretch_ascii_map(s, scale_x=2, scale_y=2),
    ),
]


def _load_map_content(ascii_map_path: str) -> str:
    with open(ascii_map_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def apply_transformations(
    directory: str,
    name: str,
    transform_set: Iterable[str] | str = "all",
) -> List[str]:
    """Generate a list of SimulationConfigs for one map with selected transforms.

    - transform_set: "all" | Iterable of families ("rotation", "mirror", "stretch")
    - transform_combo: if True, also include pairwise combinations of families (off by default)
    - include_original: include the unmodified base map
    """
    original_ascii_map = f"{directory}/{name}.map"
    original_content = _load_map_content(original_ascii_map)

    families: dict[str, list[Transform]] = {
        "rotation": ROTATIONS,
        "mirror": MIRRORS,
        "stretch": STRETCHES,
    }

    sims = []

    # Single-family transforms
    for fam, transforms in families.items():
        if transform_set != "all" and fam not in transform_set:
            continue
        for t in transforms:
            print(f"Transforming {name} {fam} {t.suffix}")
            t_suffix = t.suffix
            # transform the original map
            transformed = t.apply(original_content)

            transformed_name = f"{name}_{fam}_{t_suffix}"

            with open(f"{directory}/{transformed_name}.map", "w") as f:
                f.write(transformed)

            print(f"transformed name is {transformed_name}")

            sims.append(transformed_name)

    return sims


def rotate_ascii_map(ascii_data: str, degrees: int) -> str:
    """Rotate ASCII map by 90, 180, or 270 degrees clockwise.

    Args:
        ascii_data: Raw ASCII map string
        degrees: Rotation angle (90, 180, or 270)

    Returns:
        Rotated ASCII map string

    Note:
        All characters are preserved exactly, including:
        - Converter chains (n, m, c, R, B, G)
        - Special objects (L, F, T, S, o)
        - Agents (@, A, 1-4, p, P)
        - Altars (_, a)
    """
    if degrees not in [90, 180, 270]:
        raise ValueError(f"Degrees must be 90, 180, or 270, got {degrees}")

    lines, width, height = char_grid_to_lines(ascii_data)

    # Convert to 2D numpy array preserving all characters
    grid = np.array([list(line) for line in lines], dtype="U1")

    # Rotate using numpy (k = number of 90° rotations)
    # Use negative k for clockwise rotation (numpy default is counterclockwise)
    k = -(degrees // 90)
    rotated = np.rot90(grid, k=k)

    # Convert back to string
    rotated_lines = ["".join(row) for row in rotated]
    return "\n".join(rotated_lines)


def mirror_ascii_map(ascii_data: str, axis: Literal["horizontal", "vertical"]) -> str:
    """Mirror ASCII map along specified axis.

    Args:
        ascii_data: Raw ASCII map string
        axis: "horizontal" (flip left-right) or "vertical" (flip top-bottom)

    Returns:
        Mirrored ASCII map string

    Note:
        Preserves all special characters and their positions relative
        to the mirror axis. Converter chains remain functionally intact.
    """
    lines, width, height = char_grid_to_lines(ascii_data)

    if axis == "horizontal":
        # Flip each line left-to-right
        mirrored_lines = [line[::-1] for line in lines]
    elif axis == "vertical":
        # Flip lines top-to-bottom
        mirrored_lines = lines[::-1]
    else:
        raise ValueError(f"Axis must be 'horizontal' or 'vertical', got {axis}")

    return "\n".join(mirrored_lines)


def transform_ascii_map(
    ascii_data: str,
    rotate: Optional[int] = None,
    mirror_horizontal: bool = False,
    mirror_vertical: bool = False,
    stretch_x: int = 1,
    stretch_y: int = 1,
) -> str:
    """Apply rotation, mirroring, and/or stretching transformations to ASCII map.

    Args:
        ascii_data: Raw ASCII map string
        rotate: Optional rotation in degrees (90, 180, 270)
        mirror_horizontal: Whether to flip left-right
        mirror_vertical: Whether to flip top-bottom
        stretch_x: Horizontal stretch factor (>=1)
        stretch_y: Vertical stretch factor (>=1)

    Returns:
        Transformed ASCII map string

    Note:
        Transformations are applied in order: rotation, then mirroring, then stretching.
        This ensures predictable results when combining transformations.
    """
    result = ascii_data

    # Apply rotation first
    if rotate is not None:
        result = rotate_ascii_map(result, rotate)

    # Then apply mirroring
    if mirror_horizontal:
        result = mirror_ascii_map(result, "horizontal")
    if mirror_vertical:
        result = mirror_ascii_map(result, "vertical")

    # Finally apply stretching
    if stretch_x != 1 or stretch_y != 1:
        result = stretch_ascii_map(result, stretch_x, stretch_y)

    return result


def stretch_ascii_map(ascii_data: str, scale_x: int = 1, scale_y: int = 1) -> str:
    """Stretch an ASCII map by integer factors along X and/or Y axes.

    Rules:
    - Only duplicate empty cells and walls: '.', ' ', '#', 'W'
    - All other objects (agents, converters, special objects) are NOT duplicated.
      They appear once in the top-left cell of their scaled block; all other
      cells in that block are filled with '.' (empty).

    Args:
        ascii_data: Raw ASCII map string
        scale_x: Horizontal scale factor (>=1)
        scale_y: Vertical scale factor (>=1)

    Returns:
        The stretched ASCII map string.
    """
    if scale_x < 1 or scale_y < 1:
        raise ValueError("scale_x and scale_y must be >= 1")

    if scale_x == 1 and scale_y == 1:
        return ascii_data

    lines, width, height = char_grid_to_lines(ascii_data)

    # Characters that should be duplicated across the stretched area
    duplicate_chars = {".", " ", "#", "W"}

    stretched_lines: list[str] = []

    for line in lines:
        # Build two horizontal variants for this original row:
        # - first_row: includes original non-duplicated objects at the left-most cell
        # - other_rows: replaces non-duplicated objects with '.' across the stretched width
        row_first_parts: list[str] = []
        row_other_parts: list[str] = []

        for ch in line:
            if ch in duplicate_chars:
                tile_first = ch * scale_x
                tile_other = ch * scale_x
            else:
                # Place the original object at the top-left cell only; all other cells empty
                tile_first = ch + "." * (scale_x - 1)
                tile_other = "." * scale_x

            row_first_parts.append(tile_first)
            row_other_parts.append(tile_other)

        row_first = "".join(row_first_parts)
        row_other = "".join(row_other_parts)

        # Emit vertical copies
        stretched_lines.append(row_first)
        for _ in range(scale_y - 1):
            stretched_lines.append(row_other)

    return "\n".join(stretched_lines)
