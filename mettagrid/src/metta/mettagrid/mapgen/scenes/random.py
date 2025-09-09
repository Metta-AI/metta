import numpy as np

from metta.mettagrid.config import Config
from metta.mettagrid.mapgen.scene import Scene
from metta.mettagrid.object_types import ObjectTypes


class RandomParams(Config):
    objects: dict[str, int] = {}
    agents: int | dict[str, int] = 0
    too_many_is_ok: bool = True


class Random(Scene[RandomParams]):
    """
    Fill the grid with random symbols, based on configuration.

    This scene takes into account the existing grid content, and places objects in empty spaces only.

    MIGRATION NOTE: This scene now supports both legacy string-based grids and new int-based grids.
    The implementation automatically detects the grid format and uses appropriate operations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Detect grid format for migration compatibility
        self._grid_is_int = self.grid.dtype == np.uint8
        self._empty_value = ObjectTypes.EMPTY if self._grid_is_int else "empty"

    def render(self):
        height, width, params = self.height, self.width, self.params

        # Prepare agents based on grid format
        if isinstance(params.agents, int):
            if self._grid_is_int:
                agents = [ObjectTypes.AGENT_DEFAULT] * params.agents
            else:
                agents = ["agent.agent"] * params.agents
        elif isinstance(params.agents, dict):
            agents = []
            for agent, na in params.agents.items():
                if self._grid_is_int:
                    # Try to map agent group to type ID, fallback to default
                    try:
                        from metta.mettagrid.type_mapping import TypeMapping

                        type_mapping = TypeMapping()
                        agent_name = f"agent.{agent}"
                        type_id = type_mapping.get_type_id(agent_name)
                        agents.extend([type_id] * na)
                    except (KeyError, ImportError):
                        agents.extend([ObjectTypes.AGENT_DEFAULT] * na)
                else:
                    agents.extend([f"agent.{agent}"] * na)
        else:
            raise ValueError(f"Invalid agents: {params.agents}")

        # Find empty cells in the grid (format-aware)
        empty_mask = self.grid == self._empty_value
        empty_count = np.sum(empty_mask)
        empty_indices = np.where(empty_mask.flatten())[0]

        # Add all objects in the proper amounts to a single large array (format-aware)
        symbols = []
        if self._grid_is_int:
            # Convert object names to type IDs
            from metta.mettagrid.type_mapping import TypeMapping

            try:
                type_mapping = TypeMapping()
                for obj_name, count in params.objects.items():
                    try:
                        type_id = type_mapping.get_type_id(obj_name)
                        symbols.extend([type_id] * count)
                    except KeyError:
                        print(f"Warning: Unknown object {obj_name}, skipping")
            except ImportError:
                # Fallback if type mapping not available
                print("Warning: TypeMapping not available, using legacy approach")
                symbols = [obj_name for obj_name, count in params.objects.items() for _ in range(count)]
        else:
            # Legacy string-based approach
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

        # Shuffle the symbols (format-appropriate)
        if self._grid_is_int:
            symbols = np.array(symbols, dtype=np.uint8)
        else:
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
