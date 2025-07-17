import numpy as np

from metta.common.util.config import Config
from metta.map.scene import Scene


class RandomParams(Config):
    objects: dict[str, int] = {}
    agents: int | dict[str, int] = 0
    too_many_is_ok: bool = True


class Random(Scene[RandomParams]):
    """
    Fill the grid with random symbols, based on configuration.

    This scene takes into account the existing grid content, and places objects in empty spaces only.
    """

    def render(self):
        height, width, params = self.height, self.width, self.params

        if isinstance(params.agents, int):
            agents = ["agent.agent"] * params.agents
        elif isinstance(params.agents, dict):
            agents = ["agent." + str(agent) for agent, na in params.agents.items() for _ in range(na)]
        else:
            raise ValueError(f"Invalid agents: {params.agents}")

        # Find empty cells in the grid
        empty_mask = self.grid == "empty"
        empty_count = np.sum(empty_mask)
        empty_indices = np.where(empty_mask.flatten())[0]

        # Add all objects in the proper amounts to a single large array
        symbols = []
        for obj_name, count in params.objects.items():
            symbols.extend([obj_name] * count)
        symbols.extend(agents)

        if not params.too_many_is_ok and len(symbols) > empty_count:
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
