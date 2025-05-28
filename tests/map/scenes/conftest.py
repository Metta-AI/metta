import numpy as np
import pytest

from metta.map.node import Node
from metta.map.scene import Scene


class MockScene:
    def render(self, node):
        pass


@pytest.fixture
def scene_to_node():
    def factory(scene: Scene, shape: tuple[int, int]):
        grid = np.full(shape, "empty", dtype="<U50")
        node = Node(MockScene(), grid)
        scene.render(node)
        return node

    return factory
