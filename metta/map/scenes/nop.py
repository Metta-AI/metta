from metta.map.node import Node
from metta.map.scene import Scene, TypedChild


class Nop(Scene):
    """
    This scene doesn't do anything.
    """

    def __init__(self, children: list[TypedChild] | None = None):
        super().__init__(children=children)

    def _render(self, node: Node):
        pass
