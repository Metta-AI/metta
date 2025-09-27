from typing import Optional

import numpy as np

from mettagrid.map_builder.map_builder import GameMap, MapBuilder, MapBuilderConfig
from mettagrid.map_builder.utils import draw_border


class AssemblerMapBuilder(MapBuilder):
    class Config(MapBuilderConfig["AssemblerMapBuilder"]):
        seed: Optional[int] = None

        width: int = 10
        height: int = 10
        objects: dict[str, int] = {}
        agents: int | dict[str, int] = 0
        border_width: int = 0
        border_object: str = "wall"

    def __init__(self, config: Config):
        self._config = config
        self._rng = np.random.default_rng(self._config.seed)

    def build(self):
        # Reset RNG to ensure deterministic builds across multiple calls
        if self._config.seed is not None:
            self._rng = np.random.default_rng(self._config.seed)

        height = self._config.height
        width = self._config.width

        # Create empty grid
        grid = np.full((height, width), "empty", dtype="<U50")

        # Draw border first if needed
        if self._config.border_width > 0:
            draw_border(grid, self._config.border_width, self._config.border_object)

        # Calculate inner area where objects can be placed
        if self._config.border_width > 0:
            inner_height = max(0, height - 2 * self._config.border_width)
            inner_width = max(0, width - 2 * self._config.border_width)
        else:
            inner_height = height
            inner_width = width

        # If inner area is too small for a 1-cell padding around objects, return as is
        if inner_height < 3 or inner_width < 3:
            return GameMap(grid)

        # Prepare agent symbols (placed after objects)
        if isinstance(self._config.agents, int):
            agent_symbols = ["agent.agent"] * self._config.agents
        elif isinstance(self._config.agents, dict):
            agent_symbols = ["agent." + agent for agent, na in self._config.agents.items() for _ in range(na)]
        else:
            raise ValueError(f"Invalid agents configuration: {self._config.agents}")

        # Prepare object symbols
        object_symbols = []
        for obj_name, count in self._config.objects.items():
            object_symbols.extend([obj_name] * count)

        # Compute valid placement bounds that guarantee a 1-cell padding from any border
        top = self._config.border_width + 1
        left = self._config.border_width + 1
        bottom = height - self._config.border_width - 2
        right = width - self._config.border_width - 2

        if bottom < top or right < left:
            return GameMap(grid)

        # Keep a mask of reserved cells (objects and their padding)
        reserved = np.zeros((height, width), dtype=bool)

        # Helper to mark a 3x3 neighborhood as reserved around (i, j)
        def reserve_with_padding(i: int, j: int):
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < height and 0 <= jj < width:
                        reserved[ii, jj] = True

        # Generate all candidate centers that satisfy border padding
        cand_rows = np.arange(top, bottom + 1)
        cand_cols = np.arange(left, right + 1)
        candidates = [(i, j) for i in cand_rows for j in cand_cols]
        self._rng.shuffle(candidates)

        # Place objects greedily with padding constraint
        for symbol in object_symbols:
            placed = False
            for idx in range(len(candidates)):
                i, j = candidates[idx]
                # Check 3x3 neighborhood is unreserved
                if reserved[i - 1 : i + 2, j - 1 : j + 2].any():
                    continue
                # Place object
                grid[i, j] = symbol
                reserve_with_padding(i, j)
                # Remove this candidate to avoid reusing exact cell
                candidates.pop(idx)
                placed = True
                break
            if not placed:
                # No valid spot left; stop placing remaining objects
                break

        # Now place agents in remaining empty cells (no special padding required)
        if agent_symbols:
            empty_mask = grid == "empty"
            empty_indices = np.argwhere(empty_mask)
            if len(empty_indices) > 0:
                self._rng.shuffle(empty_indices)
                num_placeable = min(len(agent_symbols), len(empty_indices))
                for k in range(num_placeable):
                    i, j = empty_indices[k]
                    grid[i, j] = agent_symbols[k]

        return GameMap(grid)
