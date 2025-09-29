from mettagrid.config.config import Config
from mettagrid.mapgen.scene import Scene


class FillAreaParams(Config):
    value: str = "empty"


class FillArea(Scene[FillAreaParams]):
    """Fill the entire scene area with a single map value."""

    def render(self):
        self.grid[:] = self.params.value
