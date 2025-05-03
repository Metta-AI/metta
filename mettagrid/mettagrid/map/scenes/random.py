from typing import Optional

import numpy as np
from omegaconf import DictConfig

from mettagrid.map.node import Node
from mettagrid.map.scene import Scene
from mettagrid.map.utils.random import MaybeSeed


class Random(Scene):
    """
    Fill the grid with random symbols, based on configuration.

    This scene takes into account the existing grid content, and places objects in empty spaces only.
    """

    def __init__(
        self,
        objects: Optional[DictConfig | dict] = None,
        agents: int | DictConfig = 0,
        too_many_is_ok: bool = True,
        seed: MaybeSeed = None,
    ):
        super().__init__()
        self._rng = np.random.default_rng(seed)
        self._objects = objects or {}
        self._agents = agents
        self._too_many_is_ok = too_many_is_ok

    def _render(self, node: Node):
        height, width = node.height, node.width

        if isinstance(self._agents, int):
            agents = ["agent.agent"] * self._agents
        elif isinstance(self._agents, DictConfig):
            agents = ["agent." + str(agent) for agent, na in self._agents.items() for _ in range(na)]

        # Find empty cells in the grid
        empty_mask = node.grid == "empty"
        empty_count = np.sum(empty_mask)
        empty_indices = np.where(empty_mask.flatten())[0]

        # Add all objects in the proper amounts to a single large array
        symbols = []
        for obj_name, count in self._objects.items():
            symbols.extend([obj_name] * count)
        symbols.extend(agents)

        if not self._too_many_is_ok and len(symbols) > empty_count:
            raise ValueError(f"Too many objects for available empty cells: {len(symbols)} > {empty_count}")
        else:
            # everything will be filled with symbols, oh well
            symbols = symbols[:empty_count]

        if not symbols:
            return

        # Shuffle the symbols
        symbols = np.array(symbols).astype(str)
        self._rng.shuffle(symbols)

        # Shuffle the indices of empty cells
        self._rng.shuffle(empty_indices)

        # Take only as many indices as we have symbols
        selected_indices = empty_indices[: len(symbols)]

        # Create a flat copy of the grid
        flat_grid = node.grid.flatten()
        # Place symbols at the selected empty positions
        flat_grid[selected_indices] = symbols
        # Reshape back to original dimensions
        node.grid[:] = flat_grid.reshape(height, width)
