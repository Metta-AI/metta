"""Enhanced corridor builder with angle support for radial patterns."""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from metta.common.config import Config
from metta.mettagrid.mapgen.scene import Scene


@dataclass
class AngledCorridorSpec:
    """Specification for a corridor at any angle."""

    angle: float  # Angle in degrees (0=right/east, 90=up/north, 180=left/west, 270=down/south)
    center: Tuple[int, int]  # Starting point (y, x)
    length: int  # How far to extend
    thickness: int  # Corridor thickness
    bidirectional: bool = False  # If True, extend in both directions from center
    name: Optional[str] = None
    # Optional per-corridor wall ring settings (override global params)
    surround_with_walls: Optional[bool] = None
    wall_sides: Optional[List[str]] = None


class AngledCorridorBuilderParams(Config):
    """Parameters for angle-based corridor generation."""

    # List of corridors to create
    corridors: List[AngledCorridorSpec] = field(default_factory=list)

    # Additional corridors carved AFTER optional surrounding walls.
    # Useful for doorways that shouldn't get a wall ring.
    post_carve_corridors: List[AngledCorridorSpec] = field(default_factory=list)

    # How to initialize the scene area before carving corridors
    # - "wall": start fully walled, carve empty corridors (default)
    # - "empty": start empty, carve corridors, optionally surround with walls
    initial_fill: str = "wall"

    # Objects to place
    objects: Dict[str, int] = field(default_factory=dict)

    # Placement strategy
    place_at_ends: bool = True
    place_at_center: bool = False
    place_at_intersections: bool = False

    # Agent placement
    agent_position: Optional[Tuple[int, int]] = None

    # End placement preference (for place_at_ends): if True, prefer lower (larger y)
    prefer_lower_ends: bool = False

    # Whether to randomize placement order after collecting candidates
    shuffle_placements: bool = True

    # Ensure an altar is placed adjacent to the agent (used by some maps only)
    ensure_altar_near_agent: bool = False

    # Prefer ends far from map center when placing at ends
    prefer_far_from_center: bool = False

    # When initial_fill = "empty", optionally build a wall ring around carved corridors
    surround_with_walls: bool = False
    # Which sides to place wall cells around the corridor path.
    # Allowed values: "N", "S", "E", "W" (cardinals). Defaults to all.
    wall_sides: List[str] = field(default_factory=lambda: ["N", "S", "E", "W"])

    # Control center placement order (for place_at_center). If True, newest
    # centers (including post_carve_corridors) are prioritized.
    center_newest_first: bool = False

    # Place specific objects at fixed coordinates before strategy placement.
    # Example: {"altar": [(y1, x1), (y2, x2)]}
    fixed_objects: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)


class AngledCorridorBuilder(Scene[AngledCorridorBuilderParams]):
    """Build corridors at any angle with full control."""

    @classmethod
    def intrinsic_size(cls, params: AngledCorridorBuilderParams) -> tuple[int, int] | None:
        """Best-effort intrinsic size based on corridor endpoints and thickness.

        Returns (height, width) if all specs yield a finite bounding box; otherwise None.

        Notes:
        - Uses corridor centers and projected endpoints (via length and angle).
        - Adds a margin for thickness and the builder's 1-cell outer wall ring.
        - If inputs imply negative coordinates for the interior area, returns None.
        """
        if not params.corridors:
            return None

        # Track extrema across all corridors
        min_x: float | None = None
        min_y: float | None = None
        max_x: float | None = None
        max_y: float | None = None

        # Conservative margin: half thickness plus small bridging allowance
        global_margin = 0

        for spec in params.corridors:
            cy, cx = spec.center

            angle_rad = math.radians(spec.angle)
            dx = math.cos(angle_rad)
            dy = -math.sin(angle_rad)

            # Endpoints for one-way and bidirectional corridors
            endpoints: list[tuple[float, float]] = []
            endpoints.append((cy, cx))
            endpoints.append((cy + dy * spec.length, cx + dx * spec.length))
            if spec.bidirectional:
                endpoints.append((cy - dy * spec.length, cx - dx * spec.length))

            # Update global margin
            per_spec_margin = (spec.thickness // 2) + 2
            if per_spec_margin > global_margin:
                global_margin = per_spec_margin

            for ey, ex in endpoints:
                if min_x is None or ex < min_x:
                    min_x = ex
                if max_x is None or ex > max_x:
                    max_x = ex
                if min_y is None or ey < min_y:
                    min_y = ey
                if max_y is None or ey > max_y:
                    max_y = ey

        assert min_x is not None and min_y is not None and max_x is not None and max_y is not None

        # Account for corridor thickness and a 1-cell wall ring around the interior
        # Interior coordinates must be > 0 to leave the outer wall intact
        # If requested coordinates would push into the wall ring, we decline intrinsic sizing
        if min_x - global_margin < 1 or min_y - global_margin < 1:
            return None

        interior_max_x = int(math.ceil(max_x + global_margin))
        interior_max_y = int(math.ceil(max_y + global_margin))

        # Ensure room for 1-cell walls on all sides (empty cells must lie in 1..w-2, 1..h-2)
        width = interior_max_x + 2
        height = interior_max_y + 2

        # Sanity clamp to positive
        if width <= 2 or height <= 2:
            return None

        return (height, width)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.corridor_ends = []
        self.corridor_centers = []
        self.corridor_cells = []
        self.corridor_cells_by_spec: List[List[Tuple[int, int]]] = []
        self.intersections = []

    def render(self):
        """Render all specified corridors."""
        # Initialize fill
        if getattr(self.params, "initial_fill", "wall") == "empty":
            self.grid[:] = "empty"
        else:
            self.grid[:] = "wall"

        # Create each corridor and collect per-spec cells
        self.corridor_cells_by_spec = []
        for spec in self.params.corridors:
            spec_cells = self._create_angled_corridor(spec)
            self.corridor_cells_by_spec.append(spec_cells)

        # Optionally surround carved corridors with walls (useful when initial_fill="empty")
        if getattr(self.params, "surround_with_walls", False):
            self._surround_corridors_with_walls()

        # Post-carve corridors (e.g., doorways) - applied after surrounding walls
        if getattr(self.params, "post_carve_corridors", None):
            for spec in self.params.post_carve_corridors:
                self._create_angled_corridor(spec)

        # Find intersections if needed
        if self.params.place_at_intersections:
            self._find_intersections()

        # Place entities
        self._place_entities()

    def _surround_corridors_with_walls(self):
        """Place walls adjacent to corridor cells on selected sides.

        Walls are placed only on cells that are not part of the corridor path
        and lie strictly within the inner boundary (0 < y < h-1, 0 < x < w-1).
        """
        # Normalize sides
        allowed = {"N", "S", "E", "W"}
        sides = {s.upper() for s in getattr(self.params, "wall_sides", ["N", "S", "E", "W"]) if s.upper() in allowed}
        if not sides:
            return

        offsets = []
        if "N" in sides:
            offsets.append((-1, 0))
        if "S" in sides:
            offsets.append((1, 0))
        if "W" in sides:
            offsets.append((0, -1))
        if "E" in sides:
            offsets.append((0, 1))

        all_cells_set = set(self.corridor_cells)
        # Apply per-corridor overrides when provided
        for spec, cells in zip(self.params.corridors, self.corridor_cells_by_spec, strict=False):
            if spec.surround_with_walls is False:
                continue
            use_sides = sides
            if spec.wall_sides:
                use_sides = {s for s in spec.wall_sides if s in allowed}
                if not use_sides:
                    continue
            spec_offsets = []
            if "N" in use_sides:
                spec_offsets.append((-1, 0))
            if "S" in use_sides:
                spec_offsets.append((1, 0))
            if "W" in use_sides:
                spec_offsets.append((0, -1))
            if "E" in use_sides:
                spec_offsets.append((0, 1))

            cell_set = set(cells)
            for y, x in cell_set:
                for dy, dx in spec_offsets:
                    ny = y + dy
                    nx = x + dx
                    if 0 < ny < self.height - 1 and 0 < nx < self.width - 1:
                        is_corridor_cell = (ny, nx) in all_cells_set
                        if not is_corridor_cell and self.grid[ny, nx] == "empty":
                            self.grid[ny, nx] = "wall"

    def _create_angled_corridor(self, spec: AngledCorridorSpec) -> List[Tuple[int, int]]:
        """Create a corridor at the specified angle and return carved cells."""
        center_y, center_x = spec.center
        cells: List[Tuple[int, int]] = []

        # Convert angle to radians
        angle_rad = math.radians(spec.angle)

        # Calculate direction vector
        dx = math.cos(angle_rad)
        dy = -math.sin(angle_rad)  # Negative because y increases downward

        # Track the center
        self.corridor_centers.append(spec.center)

        # Create corridor in the specified direction(s)
        if spec.bidirectional:
            # Extend in both directions
            self._draw_corridor_line(center_y, center_x, dx, dy, spec.length, spec.thickness, cells)
            self._draw_corridor_line(center_y, center_x, -dx, -dy, spec.length, spec.thickness, cells)
        else:
            # Extend in one direction
            self._draw_corridor_line(center_y, center_x, dx, dy, spec.length, spec.thickness, cells)

        return cells

    def _draw_corridor_line(
        self,
        start_y: int,
        start_x: int,
        dx: float,
        dy: float,
        length: int,
        thickness: int,
        collector: Optional[List[Tuple[int, int]]] = None,
    ):
        """Draw a corridor line from start point in given direction."""
        # Calculate end point (float dir * length)
        end_x = int(round(start_x + dx * length))
        end_y = int(round(start_y + dy * length))

        # Determine first and last in-bounds points along the main Bresenham line
        start_cell_yx = None
        end_cell_yx = None
        for px, py in self._bresenham_line(start_x, start_y, end_x, end_y):
            if 0 < py < self.height - 1 and 0 < px < self.width - 1:
                if start_cell_yx is None:
                    start_cell_yx = (py, px)
                end_cell_yx = (py, px)

        # For better corridor continuity, we'll draw a thick line
        # by drawing multiple parallel lines
        if thickness == 1:
            # Draw a 4-connected (Manhattan) line. When Bresenham advances
            # diagonally, we add a bridging cell to preserve 4-connectivity.
            # To reduce visual bias, alternate bridge orientation on each
            # diagonal transition.
            line_points = self._bresenham_line(start_x, start_y, end_x, end_y)
            prev_x, prev_y = None, None
            diagonal_toggle = False
            for x, y in line_points:
                if 0 < y < self.height - 1 and 0 < x < self.width - 1:
                    self.grid[y, x] = "empty"
                    self.corridor_cells.append((y, x))
                    if collector is not None:
                        collector.append((y, x))

                if prev_x is not None and prev_y is not None:
                    # If the step was diagonal (both x and y changed), add a bridge
                    if x != prev_x and y != prev_y:
                        # Alternate bridge orientation to avoid directional bias
                        if diagonal_toggle:
                            bridge_x, bridge_y = x, prev_y
                        else:
                            bridge_x, bridge_y = prev_x, y
                        diagonal_toggle = not diagonal_toggle
                        if 0 < bridge_y < self.height - 1 and 0 < bridge_x < self.width - 1:
                            if self.grid[bridge_y, bridge_x] != "empty":
                                self.grid[bridge_y, bridge_x] = "empty"
                                self.corridor_cells.append((bridge_y, bridge_x))
                                if collector is not None:
                                    collector.append((bridge_y, bridge_x))
                prev_x, prev_y = x, y
        else:
            # Draw multiple parallel lines for thickness
            # Calculate perpendicular direction
            perp_dx = -dy
            perp_dy = dx

            # Compute symmetric offsets so that even thicknesses are balanced
            half = thickness // 2
            if thickness % 2 == 1:
                # Odd: include center line and equal offsets
                offsets = [float(t) for t in range(-half, half + 1)]
            else:
                # Even: center corridor between cells using half-offsets
                offsets = [t + 0.5 for t in range(-half, half)]

            for t in offsets:
                # Calculate offset start and end points
                offset_x = perp_dx * t
                offset_y = perp_dy * t

                line_start_x = int(round(start_x + offset_x))
                line_start_y = int(round(start_y + offset_y))
                line_end_x = int(round(end_x + offset_x))
                line_end_y = int(round(end_y + offset_y))

                # Draw this parallel line with bridging for 4-connectivity
                line_points = self._bresenham_line(line_start_x, line_start_y, line_end_x, line_end_y)
                prev_x, prev_y = None, None
                for x, y in line_points:
                    if 0 < y < self.height - 1 and 0 < x < self.width - 1:
                        self.grid[y, x] = "empty"
                        self.corridor_cells.append((y, x))
                        if collector is not None:
                            collector.append((y, x))
                    if prev_x is not None and prev_y is not None and x != prev_x and y != prev_y:
                        if abs(dx) >= abs(dy):
                            bridge_x, bridge_y = x, prev_y
                        else:
                            bridge_x, bridge_y = prev_x, y
                        if 0 < bridge_y < self.height - 1 and 0 < bridge_x < self.width - 1:
                            if self.grid[bridge_y, bridge_x] != "empty":
                                self.grid[bridge_y, bridge_x] = "empty"
                                self.corridor_cells.append((bridge_y, bridge_x))
                                if collector is not None:
                                    collector.append((bridge_y, bridge_x))
                    prev_x, prev_y = x, y

            # Fill any gaps between parallel lines for diagonal corridors
            # This ensures the corridor is fully connected
            if abs(dx - dy) > 0.1:  # If not perfectly horizontal/vertical
                self._fill_corridor_gaps(start_x, start_y, end_x, end_y, thickness)

        # Track both in-bounds corridor endpoints for potential object placement
        if start_cell_yx is not None:
            self.corridor_ends.append(start_cell_yx)
        if end_cell_yx is not None and end_cell_yx != start_cell_yx:
            self.corridor_ends.append(end_cell_yx)

    def _bresenham_line(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
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

    def _fill_corridor_gaps(self, start_x: int, start_y: int, end_x: int, end_y: int, thickness: int):
        """Fill gaps in thick diagonal corridors to ensure connectivity."""
        # Use a simple box fill approach along the main line
        main_line = self._bresenham_line(start_x, start_y, end_x, end_y)

        half_thick = thickness // 2
        for x, y in main_line:
            # Fill a small box around each point
            for dy in range(-half_thick, half_thick + 1):
                for dx in range(-half_thick, half_thick + 1):
                    fill_y = y + dy
                    fill_x = x + dx
                    if 0 < fill_y < self.height - 1 and 0 < fill_x < self.width - 1:
                        if self.grid[fill_y, fill_x] != "empty":
                            self.grid[fill_y, fill_x] = "empty"
                            self.corridor_cells.append((fill_y, fill_x))

    def _find_intersections(self):
        """Find intersection points between corridors."""
        # Count how many times each corridor cell was written
        counts: Dict[Tuple[int, int], int] = {}
        for cell in self.corridor_cells:
            counts[cell] = counts.get(cell, 0) + 1

        # Intersections are cells written 2+ times and still empty
        self.intersections = [
            (y, x)
            for (y, x), c in counts.items()
            if c >= 2 and 0 < y < self.height - 1 and 0 < x < self.width - 1 and self.grid[y, x] == "empty"
        ]

    def _place_entities(self):
        """Place agent and objects according to strategy."""
        # Place fixed objects first, so strategic placement avoids blocking them.
        if getattr(self.params, "fixed_objects", None):
            for obj_name, coords in self.params.fixed_objects.items():
                for y, x in coords:
                    if 0 < y < self.height - 1 and 0 < x < self.width - 1:
                        if self.grid[y, x] == "empty":
                            self.grid[y, x] = obj_name
        # Place agent
        if self.params.agent_position:
            y, x = self.params.agent_position
            if 0 <= y < self.height and 0 <= x < self.width:
                self.grid[y, x] = "agent.agent"
        elif self.corridor_centers:
            # Default to first center point
            center = self.corridor_centers[0]
            if self.grid[center] == "empty":
                self.grid[center] = "agent.agent"

        # Prepare placement positions
        placement_positions: List[Tuple[int, int]] = []

        if self.params.place_at_ends and self.corridor_ends:
            valid_ends = [
                pos
                for pos in self.corridor_ends
                if 0 < pos[0] < self.height - 1 and 0 < pos[1] < self.width - 1 and self.grid[pos] == "empty"
            ]
            # Optionally sort ends to prefer lower ones (to mimic maps like hard_sequence)
            if self.params.prefer_lower_ends:
                valid_ends.sort(key=lambda p: (-p[0], p[1]))
            if self.params.prefer_far_from_center:
                center_y = self.height / 2.0
                center_x = self.width / 2.0
                valid_ends.sort(key=lambda p: abs(p[0] - center_y) + abs(p[1] - center_x), reverse=True)
            placement_positions.extend(valid_ends)

        if self.params.place_at_center and self.corridor_centers:
            centers = list(self.corridor_centers)
            if getattr(self.params, "center_newest_first", False):
                centers = list(reversed(centers))
            valid_centers = [pos for pos in centers if self.grid[pos] == "empty"]
            placement_positions.extend(valid_centers)

        if self.params.place_at_intersections:
            self._find_intersections()
            if self.intersections:
                placement_positions.extend(self.intersections)

        # Remove duplicates while preserving order, then shuffle
        seen = set()
        deduped: List[Tuple[int, int]] = []
        for pos in placement_positions:
            if pos not in seen:
                deduped.append(pos)
                seen.add(pos)
        placement_positions = deduped
        if self.params.shuffle_placements:
            self.rng.shuffle(placement_positions)

        # Place objects
        for obj_name, count in self.params.objects.items():
            for _i in range(min(count, len(placement_positions))):
                if placement_positions:
                    pos = placement_positions.pop(0)
                    self.grid[pos] = obj_name

        # Optional: ensure an altar next to the agent if requested
        if (
            self.params.ensure_altar_near_agent
            and "altar" in self.params.objects
            and (self.params.agent_position or self.corridor_centers)
        ):
            ay, ax = self.params.agent_position if self.params.agent_position else self.corridor_centers[0]
            # If there is no altar adjacent, place one to the right if empty
            has_adjacent_altar = False
            for ny, nx in ((ay, ax - 1), (ay, ax + 1), (ay - 1, ax), (ay + 1, ax)):
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    if self.grid[ny, nx] == "altar":
                        has_adjacent_altar = True
                        break
            if not has_adjacent_altar:
                # Prefer placing below (towards exits), else right/left, else above
                for ny, nx in ((ay + 1, ax), (ay, ax + 1), (ay, ax - 1), (ay - 1, ax)):
                    if 0 < ny < self.height - 1 and 0 < nx < self.width - 1 and self.grid[ny, nx] == "empty":
                        self.grid[ny, nx] = "altar"
                        break


# Helper functions for easy corridor creation


def corridor(
    center: Tuple[int, int],
    angle: float,
    length: int,
    thickness: int = 3,
    bidirectional: bool = False,
    *,
    surround_with_walls: Optional[bool] = None,
    wall_sides: Optional[List[str]] = None,
) -> AngledCorridorSpec:
    """Create a corridor at any angle.

    Args:
        center: Starting point (y, x)
        angle: Direction in degrees (0=east, 90=north, 180=west, 270=south)
        length: How far to extend
        thickness: Corridor width
        bidirectional: If True, extend in both directions
    """
    return AngledCorridorSpec(
        angle=angle,
        center=center,
        length=length,
        thickness=thickness,
        bidirectional=bidirectional,
        surround_with_walls=surround_with_walls,
        wall_sides=wall_sides,
    )


def horizontal(
    y: int,
    thickness: int = 3,
    x_start: int = 1,
    x_end: Optional[int] = None,
    *,
    surround_with_walls: Optional[bool] = None,
    wall_sides: Optional[List[str]] = None,
) -> AngledCorridorSpec:
    """Convenience function for horizontal corridor (angle=0)."""
    length = (x_end if x_end else 999) - x_start
    return corridor(
        center=(y, x_start),
        angle=0,  # East/right
        length=length,
        thickness=thickness,
        bidirectional=False,
        surround_with_walls=surround_with_walls,
        wall_sides=wall_sides,
    )


def vertical(
    x: int,
    thickness: int = 3,
    y_start: int = 1,
    y_end: Optional[int] = None,
    *,
    surround_with_walls: Optional[bool] = None,
    wall_sides: Optional[List[str]] = None,
) -> AngledCorridorSpec:
    """Convenience function for vertical corridor (angle=270)."""
    length = (y_end if y_end else 999) - y_start
    return corridor(
        center=(y_start, x),
        angle=270,  # South/down
        length=length,
        thickness=thickness,
        bidirectional=False,
        surround_with_walls=surround_with_walls,
        wall_sides=wall_sides,
    )


def radial_corridors(
    center: Tuple[int, int], num_spokes: int, length: int, thickness: int = 2, start_angle: float = 0
) -> List[AngledCorridorSpec]:
    """Create radial spokes from a center point.

    Args:
        center: Center point (y, x)
        num_spokes: Number of spokes to create
        length: Length of each spoke
        thickness: Thickness of each spoke
        start_angle: Starting angle offset in degrees

    Returns:
        List of corridor specifications forming a radial pattern
    """
    corridors = []
    angle_step = 360.0 / num_spokes  # Use float division for precision

    for i in range(num_spokes):
        # Calculate angle precisely
        angle = start_angle + i * angle_step
        corridors.append(corridor(center=center, angle=angle, length=length, thickness=thickness, bidirectional=False))

    return corridors


def star_pattern(center: Tuple[int, int], num_spokes: int, length: int, thickness: int = 2) -> List[AngledCorridorSpec]:
    """Create a star pattern (bidirectional radial corridors).

    Args:
        center: Center point (y, x)
        num_spokes: Number of spokes (will create 2x corridors)
        length: Length from center to each end
        thickness: Thickness of corridors

    Returns:
        List of corridor specifications forming a star
    """
    corridors = []
    angle_step = 360.0 / num_spokes  # Use float division for precision

    for i in range(num_spokes):
        angle = i * angle_step
        corridors.append(
            corridor(
                center=center,
                angle=angle,
                length=length,
                thickness=thickness,
                bidirectional=True,  # Extend in both directions
            )
        )

    return corridors
