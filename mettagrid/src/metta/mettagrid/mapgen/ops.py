"""Operations-based map generation primitives.

This module provides a minimal set of operations for generating maps:
- DrawOp: Draw lines with thickness
- StampOp: Stamp patterns at locations

These operations can be composed, transformed, and applied to create complex maps.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import numpy.typing as npt

# Type alias for map grids
MapGrid = npt.NDArray[np.str_]


@dataclass
class DrawOp:
    """Draw a thick line from start to end."""

    start: Tuple[float, float]  # (y, x) in grid coordinates
    end: Tuple[float, float]  # (y, x) in grid coordinates
    thickness: float = 1.0
    material: str = "empty"


@dataclass
class StampOp:
    """Stamp a pattern at a location."""

    center: Tuple[float, float]  # (y, x) in grid coordinates
    pattern: npt.NDArray[np.str_]  # 2D array of materials
    rotation: float = 0.0  # Degrees clockwise


# Union type for all operations
Operation = Union[DrawOp, StampOp]


def apply_ops(grid: MapGrid, ops: List[Operation], initial_fill: str = "wall", bounds_check: bool = True) -> None:
    """Apply operations to a grid.

    Args:
        grid: The grid to modify in-place
        ops: List of operations to apply
        initial_fill: Material to fill grid with before applying ops
        bounds_check: Whether to check bounds (disable for performance)
    """
    # Initialize grid
    if initial_fill:
        grid[:] = initial_fill

    # Apply each operation
    for op in ops:
        if isinstance(op, DrawOp):
            _apply_draw_op(grid, op, bounds_check)
        elif isinstance(op, StampOp):
            _apply_stamp_op(grid, op, bounds_check)


def _apply_draw_op(grid: MapGrid, op: DrawOp, bounds_check: bool) -> None:
    """Apply a draw operation to the grid."""
    # Get line points using Bresenham
    points = _bresenham_line(
        int(round(op.start[1])), int(round(op.start[0])), int(round(op.end[1])), int(round(op.end[0]))
    )

    height, width = grid.shape

    if op.thickness == 1:
        # Simple line
        for x, y in points:
            if bounds_check:
                if 0 <= y < height and 0 <= x < width:
                    grid[y, x] = op.material
            else:
                grid[y, x] = op.material
    else:
        # Thick line - draw multiple parallel lines
        dy = op.end[0] - op.start[0]
        dx = op.end[1] - op.start[1]
        length = np.sqrt(dx * dx + dy * dy)

        if length > 0:
            # Normalize direction
            dx /= length
            dy /= length

            # Perpendicular direction
            perp_dx = -dy
            perp_dy = dx

            # Draw parallel lines for thickness
            half_thick = op.thickness / 2
            num_lines = int(np.ceil(op.thickness))

            for i in range(num_lines):
                offset = i - half_thick + 0.5
                offset_x = perp_dx * offset
                offset_y = perp_dy * offset

                # Draw offset line
                line_points = _bresenham_line(
                    int(round(op.start[1] + offset_x)),
                    int(round(op.start[0] + offset_y)),
                    int(round(op.end[1] + offset_x)),
                    int(round(op.end[0] + offset_y)),
                )

                for x, y in line_points:
                    if bounds_check:
                        if 0 <= y < height and 0 <= x < width:
                            grid[y, x] = op.material
                    else:
                        grid[y, x] = op.material


def _apply_stamp_op(grid: MapGrid, op: StampOp, bounds_check: bool) -> None:
    """Apply a stamp operation to the grid."""
    pattern = op.pattern
    if op.rotation != 0:
        # Rotate pattern
        pattern = _rotate_pattern(pattern, op.rotation)

    ph, pw = pattern.shape
    cy, cx = int(round(op.center[0])), int(round(op.center[1]))

    # Calculate top-left corner
    sy = cy - ph // 2
    sx = cx - pw // 2

    height, width = grid.shape

    # Stamp the pattern
    for py in range(ph):
        for px in range(pw):
            gy = sy + py
            gx = sx + px

            if pattern[py, px] != "":  # Skip empty cells
                if bounds_check:
                    if 0 <= gy < height and 0 <= gx < width:
                        grid[gy, gx] = pattern[py, px]
                else:
                    grid[gy, gx] = pattern[py, px]


def _bresenham_line(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    """Get all points along a line using Bresenham's algorithm."""
    points = []

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    x, y = x0, y0

    while True:
        points.append((x, y))

        if x == x1 and y == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

    return points


def _rotate_pattern(pattern: npt.NDArray[np.str_], degrees: float) -> npt.NDArray[np.str_]:
    """Rotate a pattern by the given degrees (90, 180, 270 only for now)."""
    # Normalize to 0-360
    degrees = degrees % 360

    if degrees == 0:
        return pattern
    elif degrees == 90:
        return np.rot90(pattern, k=3)  # 3 * 90 = 270 CCW = 90 CW
    elif degrees == 180:
        return np.rot90(pattern, k=2)
    elif degrees == 270:
        return np.rot90(pattern, k=1)  # 90 CCW = 270 CW
    else:
        # For now, just round to nearest 90
        # TODO: Implement proper rotation with interpolation
        nearest = round(degrees / 90) * 90
        return _rotate_pattern(pattern, nearest)


# Pattern generators (pure functions that return operations)


def line(start: Tuple[float, float], end: Tuple[float, float], thickness: float = 1, material: str = "empty") -> DrawOp:
    """Create a line drawing operation."""
    return DrawOp(start=start, end=end, thickness=thickness, material=material)


def radial_ops(
    center: Tuple[float, float], num_spokes: int, length: float, thickness: float = 1, material: str = "empty"
) -> List[DrawOp]:
    """Generate operations for a radial pattern."""
    ops = []
    for i in range(num_spokes):
        angle = i * 2 * np.pi / num_spokes
        end = (
            center[0] + length * np.sin(angle),  # y
            center[1] + length * np.cos(angle),  # x
        )
        ops.append(DrawOp(start=center, end=end, thickness=thickness, material=material))
    return ops


def grid_ops(
    origin: Tuple[float, float], rows: int, cols: int, spacing: float, thickness: float = 1, material: str = "empty"
) -> List[DrawOp]:
    """Generate operations for a grid pattern."""
    ops = []
    oy, ox = origin

    # Horizontal lines
    for i in range(rows + 1):
        y = oy + i * spacing
        ops.append(DrawOp(start=(y, ox), end=(y, ox + cols * spacing), thickness=thickness, material=material))

    # Vertical lines
    for j in range(cols + 1):
        x = ox + j * spacing
        ops.append(DrawOp(start=(oy, x), end=(oy + rows * spacing, x), thickness=thickness, material=material))

    return ops


# Transformations (pure functions that transform operations)


def offset_ops(ops: List[Operation], offset: Tuple[float, float]) -> List[Operation]:
    """Offset all operations by the given amount."""
    result = []
    for op in ops:
        if isinstance(op, DrawOp):
            result.append(
                DrawOp(
                    start=(op.start[0] + offset[0], op.start[1] + offset[1]),
                    end=(op.end[0] + offset[0], op.end[1] + offset[1]),
                    thickness=op.thickness,
                    material=op.material,
                )
            )
        elif isinstance(op, StampOp):
            result.append(
                StampOp(
                    center=(op.center[0] + offset[0], op.center[1] + offset[1]),
                    pattern=op.pattern,
                    rotation=op.rotation,
                )
            )
    return result


def surround_with_walls(ops: List[DrawOp], wall_thickness: float = 1) -> List[DrawOp]:
    """Generate wall operations that surround the given draw operations.

    This is a simplified version that adds walls around each line.
    A more sophisticated version would compute the actual boundary.
    """
    wall_ops = []

    for op in ops:
        if isinstance(op, DrawOp):
            # Calculate perpendicular direction
            dy = op.end[0] - op.start[0]
            dx = op.end[1] - op.start[1]
            length = np.sqrt(dx * dx + dy * dy)

            if length > 0:
                # Normalize
                dx /= length
                dy /= length

                # Perpendicular
                perp_dx = -dy
                perp_dy = dx

                # Offset for walls
                offset = (op.thickness / 2) + wall_thickness / 2

                # Add walls on both sides
                for side in [-1, 1]:
                    wall_offset = (perp_dy * offset * side, perp_dx * offset * side)
                    wall_ops.append(
                        DrawOp(
                            start=(op.start[0] + wall_offset[0], op.start[1] + wall_offset[1]),
                            end=(op.end[0] + wall_offset[0], op.end[1] + wall_offset[1]),
                            thickness=wall_thickness,
                            material="wall",
                        )
                    )

    return wall_ops


# Chemistry reactions (combine operations according to rules)


def react(ops1: Sequence[Operation], ops2: Sequence[Operation], reaction: str) -> List[Operation]:
    """Combine two sets of operations according to a reaction rule.

    Reactions:
    - "merge": Simply combine both sets
    - "intersection": Add stamps at intersection points
    - "parallel": Create parallel corridors
    - "subtract": Remove ops2 areas from ops1 (requires rasterization)
    """
    if reaction == "merge":
        return [*ops1, *ops2]

    elif reaction == "intersection":
        # Find where lines cross and add stamps
        inter_result: List[Operation] = [*ops1, *ops2]
        intersections = find_line_intersections(list(ops1), list(ops2))

        # Create a simple cross pattern for intersections
        cross_pattern = np.array([["", "wall", ""], ["wall", "empty", "wall"], ["", "wall", ""]], dtype=str)

        for pt in intersections:
            inter_result.append(StampOp(center=pt, pattern=cross_pattern))

        return inter_result

    elif reaction == "parallel":
        # Create parallel versions of ops2 alongside ops1
        result: List[Operation] = list(ops1)

        # For each line in ops1, add a parallel from ops2
        for i, op1 in enumerate(ops1):
            if not isinstance(op1, DrawOp) or i >= len(ops2):
                continue
            op2_any = ops2[i]
            if not isinstance(op2_any, DrawOp):
                continue
            op2 = cast(DrawOp, op2_any)
            # Calculate offset direction
            dy = op1.end[0] - op1.start[0]
            dx = op1.end[1] - op1.start[1]
            length = np.sqrt(dx * dx + dy * dy)

            if length > 0:
                # Perpendicular offset
                perp_dx = -dy / length
                perp_dy = dx / length
                offset_dist = op1.thickness + 2  # Gap of 2 cells

                offset = (perp_dy * offset_dist, perp_dx * offset_dist)

                # Create parallel line
                result.append(
                    DrawOp(
                        start=(op2.start[0] + offset[0], op2.start[1] + offset[1]),
                        end=(op2.end[0] + offset[0], op2.end[1] + offset[1]),
                        thickness=op2.thickness,
                        material=op2.material,
                    )
                )

        return result

    else:
        raise ValueError(f"Unknown reaction type: {reaction}")


def find_line_intersections(ops1: List[Operation], ops2: List[Operation]) -> List[Tuple[float, float]]:
    """Find intersection points between two sets of line operations."""
    intersections = []

    for op1 in ops1:
        if not isinstance(op1, DrawOp):
            continue

        for op2 in ops2:
            if not isinstance(op2, DrawOp):
                continue

            # Simple line intersection algorithm
            pt = line_intersection(
                op1.start[1], op1.start[0], op1.end[1], op1.end[0], op2.start[1], op2.start[0], op2.end[1], op2.end[0]
            )

            if pt is not None:
                intersections.append((pt[1], pt[0]))  # Convert back to (y, x)

    return intersections


def line_intersection(
    x1: float, y1: float, x2: float, y2: float, x3: float, y3: float, x4: float, y4: float
) -> Optional[Tuple[float, float]]:
    """Find intersection point of two line segments, if it exists."""
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if abs(denom) < 1e-10:
        return None  # Lines are parallel

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        # Intersection exists within both line segments
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return (x, y)

    return None
