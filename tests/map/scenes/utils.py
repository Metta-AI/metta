import numpy as np
import pytest

from metta.map.node import Node
from metta.map.scene import Scene
from metta.map.utils.ascii_grid import bordered_text_to_lines
from metta.map.utils.storable_map import grid_to_ascii


class MockScene(Scene):
    def _render(self, node: Node):
        pass


def scene_to_node(scene: Scene, shape: tuple[int, int]):
    grid = np.full(shape, "empty", dtype="<U50")
    node = Node(MockScene(), grid)
    scene.render(node)
    return node


def check_grid(node: Node, ascii_grid: str):
    grid_lines = grid_to_ascii(node.grid)
    expected_lines, _, _ = bordered_text_to_lines(ascii_grid)

    if grid_lines != expected_lines:
        expected_grid = "\n".join(expected_lines)
        actual_grid = "\n".join(grid_lines)
        pytest.fail(f"Grid does not match expected:\nEXPECTED:\n{expected_grid}\n\nACTUAL:\n{actual_grid}")
