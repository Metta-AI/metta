import numpy as np

from metta.common.util.config import Config
from metta.map.scene import Scene


class RandomParams(Config):
    objects: dict[str, int] = {}
    agents: int | dict[str, int] = 0
    too_many_is_ok: bool = True


class Random(Scene[RandomParams]):
    """
    Fill the perimeter of the grid with random symbols, based on configuration.

    This scene places objects only along the edges of the grid, touching the borders.
    """

    def render(self):
        height, width, params = self.height, self.width, self.params

        if isinstance(params.agents, int):
            agents = ["agent.agent"] * params.agents
        elif isinstance(params.agents, dict):
            agents = ["agent." + str(agent) for agent, na in params.agents.items() for _ in range(na)]
        else:
            raise ValueError(f"Invalid agents: {params.agents}")

        # Find perimeter cells (cells touching the border)
        perimeter_mask = np.zeros((height, width), dtype=bool)

        # Top and bottom rows
        perimeter_mask[0, :] = True
        perimeter_mask[height-1, :] = True

        # Left and right columns
        perimeter_mask[:, 0] = True
        perimeter_mask[:, width-1] = True

        # Find empty perimeter cells
        empty_perimeter_mask = (self.grid == "empty") & perimeter_mask
        empty_perimeter_count = np.sum(empty_perimeter_mask)
        empty_perimeter_indices = np.where(empty_perimeter_mask.flatten())[0]

        # Add all objects in the proper amounts to a single large array
        symbols = []
        for obj_name, count in params.objects.items():
            symbols.extend([obj_name] * count)
        symbols.extend(agents)

        if not params.too_many_is_ok and len(symbols) > empty_perimeter_count:
            raise ValueError(f"Too many objects for available perimeter cells: {len(symbols)} > {empty_perimeter_count}")
        else:
            # everything will be filled with symbols, oh well
            symbols = symbols[:empty_perimeter_count]

        if not symbols:
            return

        # Shuffle the symbols
        symbols = np.array(symbols).astype(str)
        self.rng.shuffle(symbols)

        # Shuffle the indices of empty perimeter cells
        self.rng.shuffle(empty_perimeter_indices)

        # Take only as many indices as we have symbols
        selected_indices = empty_perimeter_indices[: len(symbols)]

        # Create a flat copy of the grid
        flat_grid = self.grid.flatten()
        # Place symbols at the selected empty perimeter positions
        flat_grid[selected_indices] = symbols
        # Reshape back to original dimensions
        self.grid[:] = flat_grid.reshape(height, width)
