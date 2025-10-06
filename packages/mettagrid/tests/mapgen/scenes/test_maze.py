import pytest

from mettagrid.mapgen.random.int import IntUniformDistribution
from mettagrid.mapgen.scenes.maze import Maze
from mettagrid.test_support.mapgen import assert_connected, render_scene


@pytest.mark.parametrize("algorithm", ["kruskal", "dfs"])
def test_basic(algorithm):
    scene = render_scene(
        Maze.Config(
            algorithm=algorithm,
            room_size=IntUniformDistribution(low=1, high=1),
            wall_size=IntUniformDistribution(low=1, high=1),
        ),
        (9, 9),
    )

    assert_connected(scene.grid)

    # The number of walls is fixed for a given size.
    # For height 9, the simplest maze has 4 continuous horizontal walls, each with length 8.
    assert (scene.grid == "wall").sum() == 4 * 8


@pytest.mark.parametrize("algorithm", ["kruskal", "dfs"])
def test_uniform_distribution(algorithm):
    scene = render_scene(
        Maze.Config(
            algorithm=algorithm,
            room_size=IntUniformDistribution(low=1, high=2),
            wall_size=IntUniformDistribution(low=1, high=2),
        ),
        (15, 15),
    )

    assert_connected(scene.grid)
    assert 10 < (scene.grid == "wall").sum() < 200
