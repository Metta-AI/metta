from typing import Optional

import numpy as np

from mettagrid.map_builder.map_builder import GameMap, MapBuilder, MapBuilderConfig
from mettagrid.map_builder.utils import draw_border


class RandomMapBuilderConfig(MapBuilderConfig["RandomMapBuilder"]):
    """
    Configuration for building a random map.
    """

    seed: Optional[int] = None

    width: int = 10
    height: int = 10
    objects: dict[str, int] = {}
    agents: int | dict[str, int] = 0
    border_width: int = 0
    border_object: str = "wall"


class RandomMapBuilder(MapBuilder[RandomMapBuilderConfig]):
    def __init__(self, config: RandomMapBuilderConfig):
        super().__init__(config)
        self._rng = np.random.default_rng(self.config.seed)

    def build(self):
        # Reset RNG to ensure deterministic builds across multiple calls
        if self.config.seed is not None:
            self._rng = np.random.default_rng(self.config.seed)

        # Create empty grid
        grid = np.full((self.config.height, self.config.width), "empty", dtype="<U50")

        # Draw border first if needed
        if self.config.border_width > 0:
            draw_border(grid, self.config.border_width, self.config.border_object)

        # Calculate inner area where objects can be placed
        if self.config.border_width > 0:
            inner_height = max(0, self.config.height - 2 * self.config.border_width)
            inner_width = max(0, self.config.width - 2 * self.config.border_width)
            inner_area = inner_height * inner_width
        else:
            inner_height = self.config.height
            inner_width = self.config.width
            inner_area = self.config.width * self.config.height

        if inner_area <= 0:
            return GameMap(grid)  # No room for objects, return border-only grid

        # Prepare agent symbols
        if isinstance(self.config.agents, int):
            agents = ["agent.agent"] * self.config.agents
        elif isinstance(self.config.agents, dict):
            agents = ["agent." + agent for agent, na in self.config.agents.items() for _ in range(na)]
        else:
            raise ValueError(f"Invalid agents configuration: {self.config.agents}")

        # Check if total objects exceed inner room size and halve counts if needed
        total_objects = sum(count for count in self.config.objects.values()) + len(agents)
        while total_objects > inner_area:
            # If we can't reduce further, break to avoid infinite loop
            all_ones = all(count <= 1 for count in self.config.objects.values()) and len(agents) <= 1
            if all_ones:
                break
            for obj_name in self.config.objects:
                self.config.objects[obj_name] = max(1, self.config.objects[obj_name] // 2)
            total_objects = sum(count for count in self.config.objects.values()) + len(agents)

        # Create symbols array for inner area only
        symbols = []
        for obj_name, count in self.config.objects.items():
            symbols.extend([obj_name] * count)
        symbols.extend(agents)

        # Fill remaining inner area with empty
        symbols.extend(["empty"] * (inner_area - len(symbols)))

        # Shuffle and place in inner area
        symbols = np.array(symbols).astype(str)
        self._rng.shuffle(symbols)
        inner_grid = symbols.reshape(inner_height, inner_width)

        # Place inner grid into main grid
        if self.config.border_width > 0:
            grid[
                self.config.border_width : self.config.border_width + inner_height,
                self.config.border_width : self.config.border_width + inner_width,
            ] = inner_grid
        else:
            grid = inner_grid

        return GameMap(grid)
