from metta.map.scenes.maze import MazeKruskal
from tests.map.scenes.utils import assert_connected, render_node


def test_basic():
    node = render_node(MazeKruskal, {}, (9, 9))

    assert_connected(node.grid)

    # The number of walls is fixed for a given size.
    # For height 9, the simplest maze has 4 continuous horizontal walls, each with length 8.
    assert (node.grid == "wall").sum() == 4 * 8
