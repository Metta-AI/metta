from typing import Optional

import numpy as np

from metta.mettagrid.map_builder.map_builder import GameMap, MapBuilder, MapBuilderConfig
from metta.mettagrid.map_builder.utils import draw_border


class PerimeterInContextMapBuilder(MapBuilder):
    class Config(MapBuilderConfig["PerimeterInContextMapBuilder"]):
        """
        Configuration for building a mini in-context learning map.

        Objects appear on the perimeter, and the agent appears in the center.

        Always a single agent in this map.
        """

        seed: Optional[int] = None

        width: int = 7
        height: int = 7
        objects: dict[str, int] = {}
        agents: int | dict[str, int] = 1
        border_width: int = 0
        border_object: str = "wall"

    def __init__(self, config: Config):
        self._config = config
        self._rng = np.random.default_rng(self._config.seed)

    def build(self):
        height = self._config.height
        width = self._config.width

        # Create empty grid
        grid = np.full((height, width), "empty", dtype="<U50")

        # Draw border first if needed
        if self._config.border_width > 0:
            draw_border(grid, self._config.border_width, self._config.border_object)

        # Calculate inner area where objects can be placed
        if self._config.border_width > 0:
            inner_height = max(0, self._config.height - 2 * self._config.border_width)
            inner_width = max(0, self._config.width - 2 * self._config.border_width)
            inner_area = inner_height * inner_width
        else:
            inner_height = self._config.height
            inner_width = self._config.width
            inner_area = self._config.width * self._config.height

        if inner_area <= 0:
            return GameMap(grid)  # No room for objects, return border-only grid

        # always a single agent
        agents = ["agent.agent"]

        # Find perimeter cells (cells touching the border)
        perimeter_mask = np.zeros((height, width), dtype=bool)

        # Top and bottom rows
        perimeter_mask[0, :] = True
        perimeter_mask[height - 1, :] = True

        # Left and right columns
        perimeter_mask[:, 0] = True
        perimeter_mask[:, width - 1] = True

        # Exclude the four corners from placement
        if height >= 2 and width >= 2:
            perimeter_mask[0, 0] = False
            perimeter_mask[0, width - 1] = False
            perimeter_mask[height - 1, 0] = False
            perimeter_mask[height - 1, width - 1] = False

        # Find empty perimeter cells for objects
        empty_perimeter_mask = (grid == "empty") & perimeter_mask
        empty_perimeter_indices = np.where(empty_perimeter_mask.flatten())[0]

        # Prepare objects for perimeter placement
        object_symbols = []
        for obj_name, count in self._config.objects.items():
            object_symbols.extend([obj_name] * count)

        flat_grid = grid.flatten()

        # Place objects on perimeter
        object_symbols = np.array(object_symbols).astype(str)
        self._rng.shuffle(object_symbols)
        self._rng.shuffle(empty_perimeter_indices)
        selected_perimeter_indices = empty_perimeter_indices[: len(object_symbols)]
        flat_grid[selected_perimeter_indices] = object_symbols

        # place agent in center
        center_index = (height // 2) * width + (width // 2)
        flat_grid[center_index] = agents[0]

        grid = flat_grid.reshape(height, width)

        return GameMap(grid)
