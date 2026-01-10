"""Advanced maze navigation algorithms for large map exploration.

Implements:
- Wall-following (right-hand rule) for maze traversal
- Systematic quadrant scanning with wavefront expansion
- Spiral exploration from anchor points (chargers)
- Dead-end detection and backtracking
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from .harvest_policy import HarvestState
    from .map import MapManager, MapCellType


class WallFollowMode(Enum):
    """Wall-following direction preference."""
    RIGHT_HAND = "right"  # Follow right wall
    LEFT_HAND = "left"    # Follow left wall


class MazeNavigator:
    """Advanced maze navigation algorithms for systematic exploration."""

    def __init__(self, logger, obs_hr: int = None, obs_wr: int = None, tag_names: dict = None):
        """Initialize maze navigator.

        Args:
            logger: Logger instance
            obs_hr: Observation half-height radius (optional)
            obs_wr: Observation half-width radius (optional)
            tag_names: Tag ID to name mapping (optional)
        """
        self._logger = logger
        self._obs_hr = obs_hr
        self._obs_wr = obs_wr
        self._tag_names = tag_names or {}
        # Track wall-following state
        self._wall_follow_mode = WallFollowMode.RIGHT_HAND
        self._last_wall_follow_direction = None

    def wall_follow_next_direction(
        self,
        state: HarvestState,
        map_manager: MapManager,
        mode: WallFollowMode = WallFollowMode.RIGHT_HAND
    ) -> Optional[str]:
        """Use wall-following algorithm to navigate maze.

        Right-hand rule: Keep right hand on wall and follow it.
        This guarantees complete maze exploration (for connected mazes).

        Args:
            state: Current agent state
            map_manager: MapManager with map knowledge
            mode: Which wall to follow (right or left hand)

        Returns:
            Direction to move ("north", "south", "east", "west"), or None if stuck
        """
        from .map import MapCellType

        # Direction priority order for right-hand rule
        # If facing north: try right (east) > forward (north) > left (west) > back (south)
        current_dir = self._last_wall_follow_direction or "north"

        # Define rotation orders
        if mode == WallFollowMode.RIGHT_HAND:
            # Try: right, forward, left, back
            rotation_order = self._get_right_hand_order(current_dir)
        else:
            # Try: left, forward, right, back
            rotation_order = self._get_left_hand_order(current_dir)

        # Try each direction in priority order
        for direction in rotation_order:
            dr, dc = self._direction_to_offset(direction)
            next_r, next_c = state.row + dr, state.col + dc

            # Check if traversable in map
            if (0 <= next_r < state.map_height and
                0 <= next_c < state.map_width and
                map_manager.is_traversable(next_r, next_c)):

                # Also check if clear in observation (avoid walking into agents)
                if self._is_direction_clear_in_obs(state, direction):
                    self._last_wall_follow_direction = direction
                    self._logger.debug(f"Step {state.step_count}: WALL-FOLLOW: Moving {direction} (mode={mode.value}, from={current_dir})")
                    return direction

        # Completely stuck - no valid moves
        self._logger.warning(f"Step {state.step_count}: WALL-FOLLOW: No valid moves from {state.row},{state.col}")
        return None

    def _get_right_hand_order(self, facing: str) -> list[str]:
        """Get direction priority for right-hand wall following.

        Args:
            facing: Current facing direction

        Returns:
            List of directions in priority order [right, forward, left, back]
        """
        orders = {
            "north": ["east", "north", "west", "south"],
            "east": ["south", "east", "north", "west"],
            "south": ["west", "south", "east", "north"],
            "west": ["north", "west", "south", "east"],
        }
        return orders[facing]

    def _get_left_hand_order(self, facing: str) -> list[str]:
        """Get direction priority for left-hand wall following.

        Args:
            facing: Current facing direction

        Returns:
            List of directions in priority order [left, forward, right, back]
        """
        orders = {
            "north": ["west", "north", "east", "south"],
            "east": ["north", "east", "south", "west"],
            "south": ["east", "south", "west", "north"],
            "west": ["south", "west", "north", "east"],
        }
        return orders[facing]

    def get_systematic_exploration_target(
        self,
        state: HarvestState,
        map_manager: MapManager,
        anchor_point: tuple[int, int]
    ) -> Optional[tuple[int, int]]:
        """Find next unexplored cell using systematic wavefront expansion.

        Expands in concentric squares from anchor point, ensuring complete coverage.
        More systematic than frontier-based exploration.

        Args:
            state: Current agent state
            map_manager: MapManager with complete map grid
            anchor_point: Starting point for expansion (typically nearest charger)

        Returns:
            Position of nearest unexplored cell using wavefront, or None
        """
        from .map import MapCellType

        # Expand in concentric squares from anchor
        # Start at radius 1, expand until we find unknown cells or reach map edge
        max_radius = max(state.map_height, state.map_width) // 2

        for radius in range(1, max_radius + 1):
            # Check perimeter of square at this radius
            targets = self._get_square_perimeter(anchor_point, radius, state.map_height, state.map_width)

            # Filter to unknown cells adjacent to explored cells (frontier)
            frontier_targets = []
            for r, c in targets:
                cell_type = map_manager.grid[r][c]

                # Found unexplored area
                if cell_type == MapCellType.UNKNOWN:
                    # Check if adjacent to explored cell (this is the frontier)
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < state.map_height and
                            0 <= nc < state.map_width and
                            map_manager.grid[nr][nc] not in (MapCellType.UNKNOWN, MapCellType.WALL)):
                            # Found explored cell adjacent to UNKNOWN
                            frontier_targets.append((nr, nc))
                            break

            if frontier_targets:
                # Return nearest frontier cell by Manhattan distance
                current = (state.row, state.col)
                nearest = min(
                    frontier_targets,
                    key=lambda pos: abs(pos[0] - current[0]) + abs(pos[1] - current[1])
                )
                self._logger.debug(f"Step {state.step_count}: WAVEFRONT: Found target at {nearest} (radius={radius} from anchor {anchor_point})")
                return nearest

        # All areas explored up to max radius
        self._logger.debug(f"Step {state.step_count}: WAVEFRONT: No unexplored areas found within radius {max_radius}")
        return None

    def _get_square_perimeter(
        self,
        center: tuple[int, int],
        radius: int,
        map_height: int,
        map_width: int
    ) -> list[tuple[int, int]]:
        """Get cells on perimeter of square at given radius from center.

        Args:
            center: Center point (row, col)
            radius: Distance from center
            map_height: Map height limit
            map_width: Map width limit

        Returns:
            List of positions on square perimeter
        """
        cr, cc = center
        perimeter = []

        # Top and bottom edges
        for c in range(cc - radius, cc + radius + 1):
            # Top edge
            r_top = cr - radius
            if 0 <= r_top < map_height and 0 <= c < map_width:
                perimeter.append((r_top, c))
            # Bottom edge
            r_bottom = cr + radius
            if 0 <= r_bottom < map_height and 0 <= c < map_width:
                perimeter.append((r_bottom, c))

        # Left and right edges (excluding corners already added)
        for r in range(cr - radius + 1, cr + radius):
            # Left edge
            c_left = cc - radius
            if 0 <= r < map_height and 0 <= c_left < map_width:
                perimeter.append((r, c_left))
            # Right edge
            c_right = cc + radius
            if 0 <= r < map_height and 0 <= c_right < map_width:
                perimeter.append((r, c_right))

        return perimeter

    def find_largest_unexplored_region(
        self,
        state: HarvestState,
        map_manager: MapManager
    ) -> Optional[tuple[int, int]]:
        """Find the center of the largest contiguous unexplored region.

        Uses flood-fill to identify unexplored regions and targets the largest.
        Good for ensuring all map areas are explored.

        Args:
            state: Current agent state
            map_manager: MapManager with complete map grid

        Returns:
            Target position in largest unexplored region, or None
        """
        from .map import MapCellType

        # Flood-fill to find all unexplored regions
        visited = set()
        regions = []

        # Scan map for unexplored cells
        for r in range(state.map_height):
            for c in range(state.map_width):
                if (r, c) in visited:
                    continue

                if map_manager.grid[r][c] == MapCellType.UNKNOWN:
                    # Start flood-fill from this unknown cell
                    region = self._flood_fill_region(map_manager, r, c, visited, state.map_height, state.map_width)
                    if region:
                        regions.append(region)

        if not regions:
            return None

        # Find largest region
        largest_region = max(regions, key=len)

        # Find frontier cell closest to center of largest region
        # Center = average position of all cells in region
        avg_r = sum(r for r, c in largest_region) // len(largest_region)
        avg_c = sum(c for r, c in largest_region) // len(largest_region)

        # Find explored cell adjacent to this unknown region
        frontier_cells = []
        for r, c in largest_region:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < state.map_height and
                    0 <= nc < state.map_width and
                    map_manager.grid[nr][nc] not in (MapCellType.UNKNOWN, MapCellType.WALL)):
                    frontier_cells.append((nr, nc))

        if frontier_cells:
            # Return frontier cell closest to region center
            target = min(
                frontier_cells,
                key=lambda pos: abs(pos[0] - avg_r) + abs(pos[1] - avg_c)
            )
            self._logger.info(f"Step {state.step_count}: REGION-FINDER: Largest unexplored region has {len(largest_region)} cells, targeting {target}")
            return target

        return None

    def _flood_fill_region(
        self,
        map_manager: MapManager,
        start_r: int,
        start_c: int,
        visited: set,
        map_height: int,
        map_width: int
    ) -> list[tuple[int, int]]:
        """Flood-fill to find contiguous unexplored region.

        Args:
            map_manager: MapManager with map grid
            start_r: Starting row
            start_c: Starting column
            visited: Set of already-visited cells (will be modified)
            map_height: Map height
            map_width: Map width

        Returns:
            List of cells in this unexplored region
        """
        from .map import MapCellType

        region = []
        stack = [(start_r, start_c)]

        while stack:
            r, c = stack.pop()

            if (r, c) in visited:
                continue

            if not (0 <= r < map_height and 0 <= c < map_width):
                continue

            if map_manager.grid[r][c] != MapCellType.UNKNOWN:
                continue

            visited.add((r, c))
            region.append((r, c))

            # Add neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                stack.append((r + dr, c + dc))

        return region

    def _direction_to_offset(self, direction: str) -> tuple[int, int]:
        """Convert direction string to row/col offset.

        Args:
            direction: Direction name

        Returns:
            (row_offset, col_offset)
        """
        offsets = {
            "north": (-1, 0),
            "south": (1, 0),
            "east": (0, 1),
            "west": (0, -1),
        }
        return offsets[direction]

    def _is_direction_clear_in_obs(self, state: HarvestState, direction: str) -> bool:
        """Check if direction is clear in current observation.

        Args:
            state: Current agent state with observation
            direction: Direction to check

        Returns:
            True if clear, False if blocked
        """
        if state.current_obs is None:
            return True

        # Use configured observation dimensions
        obs_hr = self._obs_hr if self._obs_hr is not None else state.current_obs.height // 2
        obs_wr = self._obs_wr if self._obs_wr is not None else state.current_obs.width // 2

        dir_offsets = {"north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1)}
        dr, dc = dir_offsets[direction]
        target_obs_pos = (obs_hr + dr, obs_wr + dc)

        # Check all tokens at target position
        for tok in state.current_obs.tokens:
            if tok.location == target_obs_pos and tok.feature.name == "tag":
                # Check if this tag blocks movement
                tag_name = self._tag_names.get(tok.value, "").lower()
                # Block on truly impassable objects: walls and agents
                # All game objects (extractors, chargers, assemblers, chests) are traversable
                # - you move ONTO them to use them
                if "wall" in tag_name or "agent" in tag_name:
                    return False

        return True
