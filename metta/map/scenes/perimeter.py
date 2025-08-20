import numpy as np

from metta.common.util.config import Config
from metta.map.scene import Scene


class PerimeterParams(Config):
    objects: dict[str, int] = {}
    agents: int | dict[str, int] = 0
    too_many_is_ok: bool = True


class Perimeter(Scene[PerimeterParams]):
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
        empty_perimeter_mask = (self.grid == "empty") & perimeter_mask
        empty_perimeter_count = np.sum(empty_perimeter_mask)
        empty_perimeter_indices = np.where(empty_perimeter_mask.flatten())[0]

        # Find center cells for agents
        center_mask = np.zeros((height, width), dtype=bool)
        # Define center region (excluding perimeter)
        if height > 2 and width > 2:
            center_mask[1 : height - 1, 1 : width - 1] = True

        empty_center_mask = (self.grid == "empty") & center_mask
        empty_center_count = np.sum(empty_center_mask)
        empty_center_indices = np.where(empty_center_mask.flatten())[0]

        # Prepare objects for perimeter placement
        object_symbols = []
        for obj_name, count in params.objects.items():
            object_symbols.extend([obj_name] * count)

        # Check if objects fit on perimeter
        if not params.too_many_is_ok and len(object_symbols) > empty_perimeter_count:
            raise ValueError(
                f"Too many objects for available perimeter cells: {len(object_symbols)} > {empty_perimeter_count}"
            )
        else:
            object_symbols = object_symbols[:empty_perimeter_count]

        # Check if agents fit in center
        if not params.too_many_is_ok and len(agents) > empty_center_count:
            raise ValueError(f"Too many agents for available center cells: {len(agents)} > {empty_center_count}")
        else:
            agents = agents[:empty_center_count]

        # Create a flat copy of the grid
        flat_grid = self.grid.flatten()

        # Place objects on perimeter
        if object_symbols:
            object_symbols = np.array(object_symbols).astype(str)
            self.rng.shuffle(object_symbols)
            self.rng.shuffle(empty_perimeter_indices)
            selected_perimeter_indices = empty_perimeter_indices[: len(object_symbols)]
            flat_grid[selected_perimeter_indices] = object_symbols

        # Place agents in center
        if agents:
            agents = np.array(agents).astype(str)
            self.rng.shuffle(agents)
            self.rng.shuffle(empty_center_indices)
            selected_center_indices = empty_center_indices[: len(agents)]
            flat_grid[selected_center_indices] = agents

        # Reshape back to original dimensions
        self.grid[:] = flat_grid.reshape(height, width)
