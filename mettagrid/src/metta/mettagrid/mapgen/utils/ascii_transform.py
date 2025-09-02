"""ASCII map transformation utilities for rotation and mirroring.

This module provides functions to transform ASCII maps while preserving
all character semantics including converter chains, agents, and special objects.
"""

from typing import Literal, Optional

import numpy as np

from metta.mettagrid.mapgen.utils.ascii_grid import char_grid_to_lines


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
    ascii_data: str, rotate: Optional[int] = None, mirror_horizontal: bool = False, mirror_vertical: bool = False
) -> str:
    """Apply rotation and/or mirroring transformations to ASCII map.

    Args:
        ascii_data: Raw ASCII map string
        rotate: Optional rotation in degrees (90, 180, 270)
        mirror_horizontal: Whether to flip left-right
        mirror_vertical: Whether to flip top-bottom

    Returns:
        Transformed ASCII map string

    Note:
        Transformations are applied in order: rotation, then mirroring.
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

    return result
