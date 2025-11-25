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

        # Distance transform from largest component.
        distances = ndimage.distance_transform_cdt(~largest_mask, metric="taxicab").astype(np.int32)

        # For each other component, pick the closest cell and dig back along gradient.
        for cid in range(1, num + 1):
            if cid == largest_id:
                continue
            comp_mask = labels == cid
            if not comp_mask.any():
                continue
            comp_dists = distances[comp_mask]
            min_d = comp_dists.min()
            # Early out: if already touching, no dig needed.
            if min_d == 0:
                continue
            start_idx = int(np.argmin(comp_dists))
            comp_coords = np.argwhere(comp_mask)
            y, x = comp_coords[start_idx]

            # Dig path downhill until distance 0.
            while distances[y, x] > 0:
                self.grid[y, x] = "empty"
                best_d = distances[y, x] - 1
                candidates: list[Cell] = []
                if y > 0 and distances[y - 1, x] == best_d:
                    candidates.append((y - 1, x))
                if y + 1 < height and distances[y + 1, x] == best_d:
                    candidates.append((y + 1, x))
                if x > 0 and distances[y, x - 1] == best_d:
                    candidates.append((y, x - 1))
                if x + 1 < width and distances[y, x + 1] == best_d:
                    candidates.append((y, x + 1))

                if not candidates:
                    break
                y, x = candidates[int(self.rng.integers(0, len(candidates)))]

        # Final assertion: fully connected.
        labels_final, num_final = ndimage.label(
            self.grid == "empty", structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        )
        if num_final > 1:
            # Fallback: recompute distances on updated grid and bridge remaining components.
            empty = self.grid == "empty"
            labels, num = ndimage.label(empty, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
            counts = np.bincount(labels.ravel())
            counts[0] = 0
            largest_id = int(np.argmax(counts))
            largest_mask = labels == largest_id
            distances = ndimage.distance_transform_cdt(~largest_mask, metric="taxicab").astype(np.int32)

            for cid in range(1, num + 1):
                if cid == largest_id:
                    continue
                comp_mask = labels == cid
                if not comp_mask.any():
                    continue
                comp_dists = distances[comp_mask]
                if comp_dists.size == 0:
                    continue
                start_idx = int(np.argmin(comp_dists))
                comp_coords = np.argwhere(comp_mask)
                y, x = comp_coords[start_idx]
                while distances[y, x] > 0:
                    self.grid[y, x] = "empty"
                    best_d = distances[y, x] - 1
                    candidates: list[Cell] = []
                    if y > 0 and distances[y - 1, x] == best_d:
                        candidates.append((y - 1, x))
                    if y + 1 < height and distances[y + 1, x] == best_d:
                        candidates.append((y + 1, x))
                    if x > 0 and distances[y, x - 1] == best_d:
                        candidates.append((y, x - 1))
                    if x + 1 < width and distances[y, x + 1] == best_d:
                        candidates.append((y, x + 1))
                    if not candidates:
                        break
                    y, x = candidates[int(self.rng.integers(0, len(candidates)))]

        # Final assertion after fallback.
        labels_final, num_final = ndimage.label(
            self.grid == "empty", structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        )
        assert num_final == 1, "Map must end up with a single connected component"
