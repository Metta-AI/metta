from metta.map.scenes.maze import MazeKruskal
from tests.map.scenes.utils import is_connected, scene_to_node


def test_basic():
    scene = MazeKruskal()
    node = scene_to_node(scene, (9, 9))

    assert is_connected(node.grid)

    # The number of walls is fixed for a given size.
    # For height 9, the simplest maze has 4 continuous horizontal walls, each with length 8.
    assert (node.grid == "wall").sum() == 4 * 8
