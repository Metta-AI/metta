import logging
from collections import deque

import numpy as np
from scipy import ndimage

from mettagrid.mapgen.scene import Scene, SceneConfig

DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
# 4-connectivity structure for scipy.ndimage.label
STRUCTURE_4_CONNECTED = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

# Cost ratio for wall vs empty. Higher = prefer corridors more aggressively.
# With ratio 10, walking 10 empty cells equals digging through 1 wall.
WALL_COST_RATIO = 10

logger = logging.getLogger(__name__)


Cell = tuple[int, int]


class MakeConnectedConfig(SceneConfig):
    # Corner balancing: ensure roughly equal path distance from center to each corner.
    # Useful for fair gameplay when players spawn in corners.
    balance_corners: bool = False

    # Maximum allowed ratio of (furthest corner distance) / (nearest corner distance).
    # 1.0 = perfect balance required (may dig many shortcuts)
    # 1.5 = allow 50% deviation (moderate digging)
    # 2.0 = allow 100% deviation (minimal digging)
    balance_tolerance: float = 1.5

    # Maximum shortcuts to dig when balancing. Prevents runaway digging.
    max_balance_shortcuts: int = 10


class MakeConnected(Scene[MakeConnectedConfig]):
    """
    This scene makes the map connected by digging minimal tunnels.

    Uses weighted shortest-path to find paths that:
    - Prefer traversing existing corridors (low cost)
    - Only dig through walls when necessary (high cost)
    - Punch through at the thinnest wall sections

    Optionally balances corner distances so all corners are roughly equidistant
    from the map center (useful for fair multiplayer spawns).

    Algorithm: Dial's algorithm with bucket queues - O(n * K) where K = WALL_COST_RATIO.
    For typical grids this is effectively O(n), much faster than heap-based Dijkstra.
    """

    def _is_empty(self, symbol: str) -> bool:
        # TODO - treat agents as empty cells?
        return symbol == "empty"

    def render(self):
        height, width = self.grid.shape
        empty = self.grid == "empty"

        # === Phase 1: Ensure connectivity ===
        labels: np.ndarray
        labels, num = ndimage.label(empty, structure=STRUCTURE_4_CONNECTED)  # type: ignore[misc]
        if num > 1:
            self._connect_components(labels, num, empty, height, width)

        # === Phase 2: Balance corners (optional) ===
        if self.config.balance_corners:
            self._balance_corners(height, width)

    def _connect_components(self, labels: np.ndarray, num: int, empty: np.ndarray, height: int, width: int) -> None:
        """Connect all components to the largest one via minimal tunnels."""
        # Find the largest component (labels are 1-based in scipy)
        counts = np.bincount(labels.ravel())
        counts[0] = 0  # ignore background
        largest_id = int(np.argmax(counts))

        logger.debug(f"Found {num} components, largest is {largest_id}")

        # Compute weighted distances from largest component
        distances, predecessors = self._weighted_distances(labels == largest_id, empty, height, width)

        # Connect each non-largest component
        logger.debug(f"Connecting {num} components")
        for component_id in range(1, num + 1):
            if component_id == largest_id:
                continue

            # Get cells of this component
            comp_ys, comp_xs = np.where(labels == component_id)

            # Find the cell with minimum weighted distance to largest component
            comp_distances = distances[comp_ys, comp_xs]
            min_idx = int(np.argmin(comp_distances))
            start_y, start_x = int(comp_ys[min_idx]), int(comp_xs[min_idx])

            # Trace path back and only dig through walls
            self._dig_path(start_y, start_x, predecessors, empty)

        # Verify connectivity
        _, num_final = ndimage.label(self.grid == "empty", structure=STRUCTURE_4_CONNECTED)  # type: ignore[misc]
        assert num_final == 1, "Map must end up with a single connected component"

    def _weighted_distances(
        self, source_mask: np.ndarray, empty: np.ndarray, height: int, width: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Dial's algorithm: bucket-based shortest path for integer edge weights.

        Since we have only 2 costs (1 for empty, K for wall), we use K+1 circular buckets.
        This avoids heap operations entirely, giving O(n * K) complexity.

        Returns:
            distances: 2D int array of minimum cost to reach each cell
            predecessors: 2D array of (prev_y * width + prev_x), -1 for sources
        """
        K = WALL_COST_RATIO
        INF = height * width * K + 1  # Upper bound on any path cost

        # Use integer arrays for speed
        distances = np.full((height, width), INF, dtype=np.int32)
        # Encode predecessor as flat index (y * width + x), -1 for source/unvisited
        predecessors = np.full((height, width), -1, dtype=np.int32)

        # Circular bucket queues indexed by cost mod (K + 1)
        num_buckets = K + 1
        buckets: list[deque[tuple[int, int]]] = [deque() for _ in range(num_buckets)]

        # Seed with all source cells (cost 0)
        source_ys, source_xs = np.where(source_mask)
        for i in range(len(source_ys)):
            y, x = int(source_ys[i]), int(source_xs[i])
            distances[y, x] = 0
            predecessors[y, x] = -2  # Mark as source (distinct from -1 unvisited)
            buckets[0].append((y, x))

        # Process buckets in order of increasing cost
        current_cost = 0
        processed = 0
        total_cells = height * width

        while processed < total_cells:
            bucket_idx = current_cost % num_buckets
            bucket = buckets[bucket_idx]

            if not bucket:
                current_cost += 1
                if current_cost > INF:
                    break
                continue

            y, x = bucket.popleft()

            # Skip if we've already processed this cell with lower cost
            if distances[y, x] < current_cost:
                continue

            processed += 1

            # Explore neighbors
            for dy, dx in DIRECTIONS:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    # Cost: 1 for empty, K for wall
                    step_cost = 1 if empty[ny, nx] else K
                    new_cost = current_cost + step_cost

                    if new_cost < distances[ny, nx]:
                        distances[ny, nx] = new_cost
                        predecessors[ny, nx] = y * width + x
                        buckets[new_cost % num_buckets].append((ny, nx))

        return distances, predecessors

    def _dig_path(self, start_y: int, start_x: int, predecessors: np.ndarray, empty: np.ndarray) -> None:
        """
        Trace path from start back to the largest component, digging only through walls.
        """
        width = predecessors.shape[1]
        y, x = start_y, start_x

        while predecessors[y, x] >= 0:  # -2 = source, -1 = unvisited
            # Only dig if this cell is a wall
            if not empty[y, x]:
                self.grid[y, x] = "empty"

            # Decode predecessor from flat index
            prev_flat = predecessors[y, x]
            y, x = prev_flat // width, prev_flat % width

    # =========================================================================
    # Corner Balancing
    # =========================================================================

    def _balance_corners(self, height: int, width: int) -> None:
        """
        Iteratively add shortcuts to equalize path distances from center to corners.

        Strategy:
        1. Compute distances from map center to all cells
        2. Check if corners are within tolerance (max/min ratio)
        3. If not, find the wall cell that best shortcuts the path to the furthest corner
        4. Open that wall and repeat until balanced or max iterations reached
        """
        # Define the 4 corners and center
        corners = [
            (1, 1),  # top-left (avoid edge)
            (1, width - 2),  # top-right
            (height - 2, 1),  # bottom-left
            (height - 2, width - 2),  # bottom-right
        ]
        center = (height // 2, width // 2)

        for iteration in range(self.config.max_balance_shortcuts):
            empty = self.grid == "empty"

            # Compute distances from center
            center_mask = np.zeros((height, width), dtype=bool)
            center_mask[center] = True
            dist_from_center, _ = self._weighted_distances(center_mask, empty, height, width)

            # Get corner distances
            corner_dists = [int(dist_from_center[cy, cx]) for cy, cx in corners]
            min_dist = min(corner_dists)
            max_dist = max(corner_dists)

            # Check if balanced
            if min_dist == 0:
                logger.warning("Corner has zero distance from center - skipping balance")
                return

            ratio = max_dist / min_dist
            if ratio <= self.config.balance_tolerance:
                logger.debug(f"Corners balanced after {iteration} shortcuts (ratio={ratio:.2f})")
                return

            # Find the furthest corner
            furthest_idx = corner_dists.index(max_dist)
            furthest_corner = corners[furthest_idx]

            logger.debug(
                f"Balance iteration {iteration}: ratio={ratio:.2f}, furthest corner={furthest_corner} (dist={max_dist})"
            )

            # Find the best shortcut for this corner
            shortcut = self._find_best_shortcut(dist_from_center, empty, furthest_corner, height, width)

            if shortcut is None:
                logger.debug("No beneficial shortcut found - stopping balance")
                return

            shortcut_y, shortcut_x, improvement = shortcut
            logger.debug(f"Opening shortcut at ({shortcut_y}, {shortcut_x}), improvement={improvement}")
            self.grid[shortcut_y, shortcut_x] = "empty"

        logger.debug(f"Reached max shortcuts ({self.config.max_balance_shortcuts})")

    def _find_best_shortcut(
        self,
        dist_from_center: np.ndarray,
        empty: np.ndarray,
        corner: Cell,
        height: int,
        width: int,
    ) -> tuple[int, int, int] | None:
        """
        Find the wall cell that, if opened, would most reduce distance to the given corner.

        Returns (y, x, improvement) or None if no beneficial shortcut exists.

        The algorithm:
        1. Run Dijkstra from the corner to get distances from corner to all cells
        2. For each wall with ≥2 empty neighbors:
           - Estimate new path cost = (dist to wall from center) + 1 + (dist from wall to corner)
           - Improvement = current_dist - new_path_cost
        3. Return the wall with maximum improvement
        """
        corner_y, corner_x = corner
        current_dist = int(dist_from_center[corner_y, corner_x])

        # Compute distances from the corner
        corner_mask = np.zeros((height, width), dtype=bool)
        corner_mask[corner_y, corner_x] = True
        dist_from_corner, _ = self._weighted_distances(corner_mask, empty, height, width)

        best_cell: tuple[int, int] | None = None
        best_improvement = 0

        for y in range(height):
            for x in range(width):
                if empty[y, x]:
                    continue  # Only consider walls

                # Collect empty neighbors
                empty_neighbors: list[Cell] = []
                for dy, dx in DIRECTIONS:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and empty[ny, nx]:
                        empty_neighbors.append((ny, nx))

                # Need at least 2 empty neighbors to create a useful shortcut
                # (otherwise we're just making a dead-end accessible)
                if len(empty_neighbors) < 2:
                    continue

                # Best path through this cell if opened:
                # = (center → nearest neighbor) + 1 (enter cell) + (nearest neighbor → corner)
                best_entry = min(dist_from_center[ny, nx] for ny, nx in empty_neighbors)
                best_exit = min(dist_from_corner[ny, nx] for ny, nx in empty_neighbors)

                new_dist = best_entry + 1 + best_exit
                improvement = current_dist - new_dist

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_cell = (y, x)

        if best_cell is None:
            return None
        return (best_cell[0], best_cell[1], best_improvement)
