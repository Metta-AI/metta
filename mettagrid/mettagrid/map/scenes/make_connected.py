import logging
from typing import List, Optional

import numpy as np

from mettagrid.map.node import Node
from mettagrid.map.scene import Scene, TypedChild
from mettagrid.map.utils.random import MaybeSeed

DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

logger = logging.getLogger(__name__)


Cell = tuple[int, int]


class MakeConnected(Scene):
    """
    This scene makes the map connected by digging tunnels.

    It does this by:
    - Finding all the connected components
    - Digging shortest tunnels from the largest component to all other components

    TODO: This can result in some extra tunnels being dug.
    """

    def __init__(self, seed: MaybeSeed = None, children: Optional[List[TypedChild]] = None):
        super().__init__(children=children)
        self._rng = np.random.default_rng(seed)

    def _is_empty(self, symbol: str) -> bool:
        return symbol == "empty"

    def _render(self, node: Node):
        height, width = node.grid.shape

        component_cells = self._make_components(node)
        component_sizes = [len(cells) for cells in component_cells]

        if len(component_sizes) == 1:
            logger.debug("Map is already connected")
            return

        # find the largest component
        largest_component_id = max(range(len(component_sizes)), key=component_sizes.__getitem__)
        logger.debug(f"Largest component: {largest_component_id}")

        logger.debug("Populating distance to largest component")
        distances_to_largest_component = self._distance_to_component(
            node,
            component_cells[largest_component_id],
        )

        # connect the largest component all other components
        logger.info(f"Connecting {len(component_sizes)} components")
        for component_id, component in enumerate(component_cells):
            if component_id == largest_component_id:
                continue

            # find the cell that's closest to the largest component
            min_distance_cell = min(component, key=lambda c: distances_to_largest_component[*c])

            # shouldn't happen
            if min_distance_cell is None:
                raise ValueError("No cell found for component")

            # connect the cell to the largest component by digging a tunnel based on the shortest path
            current_cell = min_distance_cell
            current_distance = distances_to_largest_component[*current_cell]
            while current_distance > 0:
                y, x = current_cell

                # Find all neighbors with the minimum distance to the largest component
                candidates: list[Cell] = [
                    (y + dy, x + dx)
                    for dy, dx in DIRECTIONS
                    if 0 <= y + dy < height
                    and 0 <= x + dx < width
                    and distances_to_largest_component[y + dy, x + dx] == current_distance - 1
                ]

                # Pick a random candidate from those with the minimum distance
                if not candidates:
                    # This shouldn't happen if distances are calculated correctly
                    raise ValueError("No next cell found")

                next_cell = self._rng.choice(candidates)
                current_cell = next_cell
                current_distance -= 1
                node.grid[*current_cell] = "empty"

        assert len(self._make_components(node)) == 1

    def _make_components(self, node: Node):
        # run BFS from each empty cell, find connected components
        height, width = node.grid.shape

        visited = np.full((height, width), False)
        component_id = 0
        components_cells: list[list[Cell]] = []

        logger.debug("Finding components")
        for y in range(height):
            for x in range(width):
                if not self._is_empty(node.grid[y, x]):
                    continue

                # already visited
                if visited[y, x]:
                    continue

                components_cells.append([])
                queue = [(y, x)]
                i = 0
                while i < len(queue):
                    y0, x0 = queue[i]
                    i += 1
                    if visited[y0, x0]:
                        continue

                    visited[y0, x0] = True
                    components_cells[component_id].append((y0, x0))

                    for dy, dx in DIRECTIONS:
                        y1, x1 = y0 + dy, x0 + dx
                        if (
                            0 <= y1 < height
                            and 0 <= x1 < width
                            and self._is_empty(node.grid[y1, x1])
                            and not visited[y1, x1]
                        ):
                            queue.append((y1, x1))

                component_id += 1

        logger.debug(f"Found {len(components_cells)} components")
        return components_cells

    def _distance_to_component(
        self,
        node: Node,
        component_cells: list[Cell],
    ):
        height, width = node.grid.shape
        # find the distance from the component to all other cells (ignoring the occupied cells - used for finding
        # the optimal tunnels)
        distances = np.full((height, width), np.inf)
        queue = []
        for cell in component_cells:
            distances[*cell] = 0
            queue.append(cell)

        i = 0
        while i < len(queue):
            y, x = queue[i]
            i += 1

            for dy, dx in DIRECTIONS:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and distances[ny, nx] == np.inf:
                    distances[ny, nx] = distances[y, x] + 1
                    queue.append((ny, nx))

        return distances
