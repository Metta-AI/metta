from metta.mettagrid.config import Config
from metta.mettagrid.mapgen.scene import Scene


class NopParams(Config):
    pass


class Nop(Scene[NopParams]):
    """
    This scene doesn't do anything.
    """

    def render(self):
        pass
