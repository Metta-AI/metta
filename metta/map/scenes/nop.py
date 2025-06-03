from metta.map.scene import Scene
from metta.util.config import Config


class NopParams(Config):
    pass


class Nop(Scene[NopParams]):
    """
    This scene doesn't do anything.
    """

    Params = NopParams

    def render(self):
        pass
