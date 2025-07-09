import pytest

from metta.map.scenes.maze import Maze
from tests.map.scenes.utils import assert_connected, render_scene


@pytest.mark.parametrize("algorithm", ["kruskal", "dfs"])
def test_basic(algorithm):
    scene = render_scene(Maze, {"algorithm": algorithm, "room_size": 1, "wall_size": 1}, (9, 9))

    assert_connected(scene.grid)

    # The number of walls is fixed for a given size.
    # For height 9, the simplest maze has 4 continuous horizontal walls, each with length 8.
    assert (scene.grid == "wall").sum() == 4 * 8


@pytest.mark.parametrize("algorithm", ["kruskal", "dfs"])
def test_uniform_distribution(algorithm):
    scene = render_scene(
        Maze, {"algorithm": algorithm, "room_size": ("uniform", 1, 2), "wall_size": ("uniform", 1, 2)}, (15, 15)
    )

    assert_connected(scene.grid)
    assert 10 < (scene.grid == "wall").sum() < 200
