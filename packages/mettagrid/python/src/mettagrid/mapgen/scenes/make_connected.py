import logging

import numpy as np

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

        def expand(mask: np.ndarray) -> np.ndarray:
            up = np.zeros_like(mask, dtype=bool)
            down = np.zeros_like(mask, dtype=bool)
            left = np.zeros_like(mask, dtype=bool)
            right = np.zeros_like(mask, dtype=bool)
            up[:-1] = mask[1:]
            down[1:] = mask[:-1]
            left[:, :-1] = mask[:, 1:]
            right[:, 1:] = mask[:, :-1]
            return up | down | left | right

        components: list[np.ndarray] = []
        visited = np.zeros_like(empty, dtype=bool)

        # Component labeling via frontier dilation.
        while True:
            remaining = (~visited) & empty
            if not remaining.any():
                break
            y0, x0 = np.argwhere(remaining)[0]
            frontier = np.zeros_like(empty, dtype=bool)
            component_mask = np.zeros_like(empty, dtype=bool)
            frontier[y0, x0] = True
            component_mask[y0, x0] = True
            visited[y0, x0] = True

            while frontier.any():
                frontier = expand(frontier) & empty & (~visited)
                if not frontier.any():
                    break
                visited |= frontier
                component_mask |= frontier

            components.append(component_mask)

        if len(components) <= 1:
            logger.debug("Map is already connected")
            return

        sizes = [int(comp.sum()) for comp in components]
        largest_idx = int(np.argmax(sizes))
        largest = components[largest_idx]

        # Distance transform from largest component using BFS layers.
        distances = np.full_like(empty, np.inf, dtype=np.float32)
        frontier = largest.copy()
        seen = largest.copy()
        distances[frontier] = 0.0
        dist_val = 0.0

        while frontier.any():
            dist_val += 1.0
            frontier = expand(frontier) & (~seen)
            if not frontier.any():
                break
            distances[frontier] = dist_val
            seen |= frontier

        for idx, component in enumerate(components):
            if idx == largest_idx:
                continue

            comp_dist = distances[component]
            if not np.isfinite(comp_dist).any():
                continue
            flat_idx = int(np.argmin(comp_dist))
            comp_coords = np.argwhere(component)
            start_y, start_x = comp_coords[flat_idx]

            # Trace shortest path downhill in distance grid.
            y, x = int(start_y), int(start_x)
            while distances[y, x] > 0:
                self.grid[y, x] = "empty"
                # Candidate neighbors with minimal distance
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
                    # Shouldn't happen if distances are consistent.
                    break

                y, x = candidates[int(self.rng.integers(0, len(candidates)))]

        # Final assertion: fully connected.
        def _count_components(mask: np.ndarray) -> int:
            seen = np.zeros_like(mask, dtype=bool)
            count = 0
            while True:
                remaining = (~seen) & mask
                if not remaining.any():
                    break
                count += 1
                y_s, x_s = np.argwhere(remaining)[0]
                front = np.zeros_like(mask, dtype=bool)
                front[y_s, x_s] = True
                seen[y_s, x_s] = True
                while front.any():
                    front = expand(front) & mask & (~seen)
                    seen |= front
            return count

        assert _count_components(self.grid == "empty") == 1, "Map must end up with a single connected component"
