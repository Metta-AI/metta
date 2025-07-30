from metta.common.util.config import Config
from metta.map.scene import Scene


class NopParams(Config):
    pass


class Nop(Scene[NopParams]):
    """
    This scene doesn't do anything.
    """

    def render(self):
        pass
