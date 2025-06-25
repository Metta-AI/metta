from metta.map.scene import Scene
from common.src.metta.util.config import Config


class NopParams(Config):
    pass


class Nop(Scene[NopParams]):
    """
    This scene doesn't do anything.
    """

    def render(self):
        pass
