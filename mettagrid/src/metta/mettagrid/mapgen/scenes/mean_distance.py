import numpy as np

from metta.mettagrid.config import Config
from metta.mettagrid.mapgen.scene import Scene
from metta.mettagrid.object_types import ObjectTypes


class MeanDistanceParams(Config):
    mean_distance: float  # Mean distance parameter for objects relative to agent.
    objects: dict[str, int]


class MeanDistance(Scene[MeanDistanceParams]):
    """
    This scene places an agent at the center of the scene and places objects at a mean distance from the agent.

    MIGRATION NOTE: Updated to support both legacy string-based grids and new int-based grids.
    The implementation automatically detects the grid format and uses appropriate operations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Detect grid format for migration compatibility
        self._grid_is_int = self.grid.dtype == np.uint8
        self._empty_value = ObjectTypes.EMPTY if self._grid_is_int else "empty"

    def render(self):
        # Define the agent's initial position (here: center of the room)
        agent_pos = (self.height // 2, self.width // 2)

        # Place the agent at the center (format-aware)
        if self._grid_is_int:
            self.grid[agent_pos] = ObjectTypes.AGENT_DEFAULT
        else:
            self.grid[agent_pos] = "agent.agent"

        # Place each object based on a Poisson-distributed distance from the agent.
        # For each object type and the number of instances required:
        for obj_name, count in self.params.objects.items():
            placed = 0
            while placed < count:
                # Sample a distance from a Poisson distribution.
                d = self.rng.poisson(lam=self.params.mean_distance)
                # Ensure a nonzero distance (so objects don't collide with the agent)
                if d == 0:
                    d = 1
                # Sample an angle uniformly from 0 to 2*pi.
                angle = self.rng.uniform(0, 2 * np.pi)
                # Convert polar coordinates to grid offsets.
                dx = int(round(d * np.cos(angle)))
                dy = int(round(d * np.sin(angle)))
                # Candidate position (note: grid indexing is row, col so we add dy then dx).
                candidate = (agent_pos[0] + dy, agent_pos[1] + dx)
                # Check if candidate position is inside the room and unoccupied (format-aware)
                if (
                    0 <= candidate[0] < self.height
                    and 0 <= candidate[1] < self.width
                    and self.grid[candidate] == self._empty_value
                ):
                    # Place object using appropriate format
                    if self._grid_is_int:
                        try:
                            from metta.mettagrid.type_mapping import TypeMapping

                            type_mapping = TypeMapping()
                            type_id = type_mapping.get_type_id(obj_name)
                            self.grid[candidate] = type_id
                        except (KeyError, ImportError):
                            print(f"Warning: Unknown object {obj_name}, skipping placement")
                            continue
                    else:
                        self.grid[candidate] = obj_name
                    placed += 1
