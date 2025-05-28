import numpy as np

from metta.map.node import Node
from metta.map.scene import Scene


class MockScene(Scene):
    def _render(self, node: Node):
        pass


def scene_to_node(scene: Scene, shape: tuple[int, int]):
    grid = np.full(shape, "empty", dtype="<U50")
    node = Node(MockScene(), grid)
    scene.render(node)
    return node
