import numpy as np

from metta.mettagrid.config import Config
from metta.mettagrid.mapgen.scene import Scene
from metta.mettagrid.object_types import ObjectTypes


class RemoveAgentsParams(Config):
    pass


class RemoveAgents(Scene[RemoveAgentsParams]):
    """
    This class solves a frequent problem: `game.num_agents` must match the
    number of agents in the map.

    You can use this scene to remove agents from the map. Then apply `Random`
    scene to place as many agents as you want.

    (TODO - it might be better to remove `game.num_agents` from the config
    entirely, and just use the number of agents in the map.)

    MIGRATION NOTE: This scene now supports both legacy string-based grids and new int-based grids.
    The implementation automatically detects the grid format and uses appropriate operations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Detect grid format for migration compatibility
        self._grid_is_int = self.grid.dtype == np.uint8
        self._empty_value = ObjectTypes.EMPTY if self._grid_is_int else "empty"

    def render(self):
        for i in range(self.height):
            for j in range(self.width):
                value = self.grid[i, j]
                if self._grid_is_int:
                    # For int-based grids, check if it's any agent type_id
                    if ObjectTypes.is_agent(value):
                        self.grid[i, j] = self._empty_value
                else:
                    # For string-based grids, check string patterns
                    if value.startswith("agent.") or value == "agent":
                        self.grid[i, j] = self._empty_value
