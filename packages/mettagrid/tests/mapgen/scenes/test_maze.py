import pytest

import mettagrid.mapgen.random.int
import mettagrid.mapgen.scenes.maze
import mettagrid.test_support.mapgen


@pytest.mark.parametrize("algorithm", ["kruskal", "dfs"])
def test_basic(algorithm):
    scene = mettagrid.test_support.mapgen.render_scene(
        mettagrid.mapgen.scenes.maze.Maze.Config(
            algorithm=algorithm,
            room_size=mettagrid.mapgen.random.int.IntUniformDistribution(low=1, high=1),
            wall_size=mettagrid.mapgen.random.int.IntUniformDistribution(low=1, high=1),
        ),
        (9, 9),
    )

    mettagrid.test_support.mapgen.assert_connected(scene.grid)

    # The number of walls is fixed for a given size.
    # For height 9, the simplest maze has 4 continuous horizontal walls, each with length 8.
    assert (scene.grid == "wall").sum() == 4 * 8


@pytest.mark.parametrize("algorithm", ["kruskal", "dfs"])
def test_uniform_distribution(algorithm):
    scene = mettagrid.test_support.mapgen.render_scene(
        mettagrid.mapgen.scenes.maze.Maze.Config(
            algorithm=algorithm,
            room_size=mettagrid.mapgen.random.int.IntUniformDistribution(low=1, high=2),
            wall_size=mettagrid.mapgen.random.int.IntUniformDistribution(low=1, high=2),
        ),
        (15, 15),
    )

    mettagrid.test_support.mapgen.assert_connected(scene.grid)
    assert 10 < (scene.grid == "wall").sum() < 200
