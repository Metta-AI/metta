from mettagrid.config.config import Config
from mettagrid.mapgen.scene import Scene


class NopParams(Config):
    pass


class Nop(Scene[NopParams]):
    """
    This scene doesn't do anything.
    """

    def render(self):
        pass
