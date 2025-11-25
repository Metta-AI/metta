import logging

import numpy as np
from scipy import ndimage

from mettagrid.mapgen.scene import Scene, SceneConfig

DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

logger = logging.getLogger(__name__)


Cell = tuple[int, int]


class MakeConnectedConfig(SceneConfig):
    pass


class MakeConnected(Scene[MakeConnectedConfig]):
    """
    This scene makes the map connected by digging tunnels.

    It does this by:
    - Finding all the connected components
    - Digging shortest tunnels from the largest component to all other components

    TODO: This can result in some extra tunnels being dug.
    """

    def _is_empty(self, symbol: str) -> bool:
        # TODO - treat agents as empty cells?
        return symbol == "empty"

    def render(self):
        height, width = self.grid.shape
        empty = self.grid == "empty"

        # Label components using SciPy; 4-connectivity matches prior logic.
        labels, num = ndimage.label(empty, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
        if num <= 1:
            logger.debug("Map is already connected")
            return

        # Largest component id (1-based in scipy label output).
        counts = np.bincount(labels.ravel())
        counts[0] = 0  # background
        largest_id = int(np.argmax(counts))
        largest_mask = labels == largest_id

        # Distance transform (Euclidean) with nearest-point indices for path targets.
        distances, (ty, tx) = ndimage.distance_transform_edt(~largest_mask, return_indices=True, sampling=(1.0, 1.0))

        # For each other component, pick its min-distance cell in one call.
        mins = ndimage.minimum(distances, labels=labels, index=list(range(1, num + 1)))
        positions = ndimage.minimum_position(distances, labels=labels, index=list(range(1, num + 1)))

        for cid, min_d, pos in zip(range(1, num + 1), mins, positions, strict=True):
            if cid == largest_id or min_d is None:
                continue
            if min_d == 0:
                continue
            y, x = pos
            dest_y, dest_x = int(ty[y, x]), int(tx[y, x])

            # Dig a Manhattan path toward the nearest largest-component cell.
            while (y, x) != (dest_y, dest_x):
                self.grid[y, x] = "empty"
                if y != dest_y:
                    y += 1 if dest_y > y else -1
                elif x != dest_x:
                    x += 1 if dest_x > x else -1
            self.grid[dest_y, dest_x] = "empty"

        # Final assertion: fully connected.
        labels_final, num_final = ndimage.label(
            self.grid == "empty", structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        )
        assert num_final == 1, "Map must end up with a single connected component"
