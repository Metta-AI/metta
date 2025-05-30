from metta.map.node import Node
from metta.util.config import Config


class NopParams(Config):
    pass


class Nop(Node[NopParams]):
    """
    This node doesn't do anything.
    """

    params_type = NopParams

    def render(self):
        pass
