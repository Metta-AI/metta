from metta.map.scenes.maze import MazeKruskal
from tests.map.scenes.utils import assert_connected, render_scene


def test_basic():
    scene = render_scene(MazeKruskal, {"room_size": 1, "wall_size": 1}, (9, 9))

    assert_connected(scene.grid)

    # The number of walls is fixed for a given size.
    # For height 9, the simplest maze has 4 continuous horizontal walls, each with length 8.
    assert (scene.grid == "wall").sum() == 4 * 8


def test_uniform_distribution():
    scene = render_scene(MazeKruskal, {"room_size": ("uniform", 1, 2), "wall_size": ("uniform", 1, 2)}, (15, 15))

    assert_connected(scene.grid)
    assert 10 < (scene.grid == "wall").sum() < 100
