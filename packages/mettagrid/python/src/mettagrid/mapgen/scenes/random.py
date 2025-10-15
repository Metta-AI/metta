import numpy as np

from mettagrid.mapgen.scene import Scene, SceneConfig


class RandomConfig(SceneConfig):
    objects: dict[str, int] = {}
    agents: int | dict[str, int] = 0
    too_many_is_ok: bool = True


class Random(Scene[RandomConfig]):
    """
    Fill the grid with random symbols, based on configuration.

    This scene takes into account the existing grid content, and places objects in empty spaces only.
    """

    def render(self):
        height, width, config = self.height, self.width, self.config

        if isinstance(config.agents, int):
            agents = ["agent.agent"] * config.agents
        elif isinstance(config.agents, dict):
            agents = ["agent." + str(agent) for agent, na in config.agents.items() for _ in range(na)]
        else:
            raise ValueError(f"Invalid agents: {config.agents}")

        # Find empty cells in the grid
        empty_mask = self.grid == "empty"
        empty_count = np.sum(empty_mask)
        empty_indices = np.where(empty_mask.flatten())[0]

        # Add all objects in the proper amounts to a single large array
        symbols = []
        for obj_name, count in config.objects.items():
            symbols.extend([obj_name] * count)
        symbols.extend(agents)

        if not config.too_many_is_ok and len(symbols) > empty_count:
            raise ValueError(f"Too many objects for available empty cells: {len(symbols)} > {empty_count}")
        else:
            # everything will be filled with symbols, oh well
            symbols = symbols[:empty_count]

        if not symbols:
            return

        # Shuffle the symbols
        symbols = np.array(symbols).astype(str)
        self.rng.shuffle(symbols)

        # Shuffle the indices of empty cells
        self.rng.shuffle(empty_indices)

        # Take only as many indices as we have symbols
        selected_indices = empty_indices[: len(symbols)]

        # Create a flat copy of the grid
        flat_grid = self.grid.flatten()
        # Place symbols at the selected empty positions
        flat_grid[selected_indices] = symbols
        # Reshape back to original dimensions
        self.grid[:] = flat_grid.reshape(height, width)
