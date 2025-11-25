import logging

import numpy as np
import scipy.ndimage

from mettagrid.mapgen.scene import Scene, SceneConfig

DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

logger = logging.getLogger(__name__)


class MakeConnectedConfig(SceneConfig):
    pass


class MakeConnected(Scene[MakeConnectedConfig]):
    """
    This scene makes the map connected by digging tunnels.

    It does this by:
    - Finding all the connected components
    - Digging shortest tunnels from the largest component to all other components
    """

    def _is_empty(self, symbol: str) -> bool:
        # TODO - treat agents as empty cells?
        return symbol == "empty"

    def render(self):
        # Identify empty cells
        empty_mask = self.grid == "empty"

        # 4-connectivity structure for Manhattan distance logic
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

        labels, num_components = scipy.ndimage.label(empty_mask, structure=structure)

        if num_components <= 1:
            logger.debug("Map is already connected")
            return

        # find the largest component
        sizes = scipy.ndimage.sum(empty_mask, labels, range(1, num_components + 1))
        # argmax returns 0-based index, labels are 1-based
        largest_label = int(np.argmax(sizes)) + 1
        logger.debug(f"Largest component: {largest_label} (size {sizes[largest_label - 1]})")

        logger.debug("Populating distance to largest component")
        # Prepare grid for distance transform: 0 at target (largest component), 1 elsewhere
        input_grid = np.ones_like(labels, dtype=np.int32)
        input_grid[labels == largest_label] = 0

        # Calculate Manhattan (taxicab) distance from the largest component to everywhere else
        distances = scipy.ndimage.distance_transform_cdt(input_grid, metric="taxicab")

        # connect the largest component all other components
        logger.debug(f"Connecting {num_components} components")

        height, width = self.grid.shape

        for component_label in range(1, num_components + 1):
            if component_label == largest_label:
                continue

            # find the cell in this component that's closest to the largest component
            # minimum_position returns (y, x) for the first minimum found
            min_pos = scipy.ndimage.minimum_position(distances, labels, index=component_label)

            # min_pos is a tuple (y, x)
            if min_pos is None:
                # Should not happen if component exists
                raise ValueError(f"No cell found for component {component_label}")

            # connect the cell to the largest component by digging a tunnel based on the shortest path
            current_cell = min_pos
            current_distance = distances[current_cell]

            while current_distance > 0:
                y, x = current_cell

                # Find all neighbors with the minimum distance to the largest component (dist - 1)
                candidates = []
                for dy, dx in DIRECTIONS:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        if distances[ny, nx] == current_distance - 1:
                            candidates.append((ny, nx))

                if not candidates:
                    # This shouldn't happen if distances are calculated correctly
                    raise ValueError("No next cell found")

                next_cell = self.rng.choice(candidates)
                current_cell = next_cell
                current_distance -= 1

                # Mark as empty (dig)
                self.grid[current_cell] = "empty"

        # Verification
        labels_final, num_final = scipy.ndimage.label(self.grid == "empty", structure=structure)
        assert num_final == 1, f"Map must end up with a single connected component, got {num_final}"
