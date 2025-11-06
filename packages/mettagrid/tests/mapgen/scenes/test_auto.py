import numpy as np
import pytest

import mettagrid.mapgen.random.float
import mettagrid.mapgen.random.int
import mettagrid.mapgen.scenes.auto
import mettagrid.mapgen.scenes.maze
import mettagrid.mapgen.scenes.random_scene
import mettagrid.test_support.mapgen


@pytest.fixture
def common_params() -> mettagrid.mapgen.scenes.auto.AutoConfig:
    return mettagrid.mapgen.scenes.auto.AutoConfig(
        num_agents=4,
        objects={"altar": mettagrid.mapgen.random.float.FloatConstantDistribution(value=0.02)},
        room_objects={"altar": mettagrid.mapgen.random.float.FloatUniformDistribution(low=0.0005, high=0.01)},
        room_symmetry=mettagrid.mapgen.scenes.auto.AutoConfigRoomSymmetry(horizontal=1, vertical=1, x4=1, none=1),
        layout=mettagrid.mapgen.scenes.auto.AutoConfigLayout(grid=1, bsp=1),
        grid=mettagrid.mapgen.scenes.auto.AutoConfigGrid(
            rows=mettagrid.mapgen.random.int.IntConstantDistribution(value=3),
            columns=mettagrid.mapgen.random.int.IntConstantDistribution(value=3),
        ),
        bsp=mettagrid.mapgen.scenes.auto.AutoConfigBSP(
            area_count=mettagrid.mapgen.random.int.IntConstantDistribution(value=3)
        ),
        content=[
            mettagrid.mapgen.scenes.random_scene.RandomSceneCandidate(
                scene=mettagrid.mapgen.scenes.maze.Maze.Config(
                    room_size=mettagrid.mapgen.random.int.IntUniformDistribution(low=1, high=2),
                    wall_size=mettagrid.mapgen.random.int.IntUniformDistribution(low=1, high=2),
                ),
                weight=3,
            ),
        ],
    )


def test_basic(common_params):
    scene = mettagrid.test_support.mapgen.render_scene(common_params, (10, 10))

    mettagrid.test_support.mapgen.assert_connected(scene.grid)


def test_seed(common_params):
    scene1v1 = mettagrid.test_support.mapgen.render_scene(common_params.model_copy(update={"seed": 42}), (16, 16))
    scene1v2 = mettagrid.test_support.mapgen.render_scene(common_params.model_copy(update={"seed": 42}), (16, 16))
    scene2 = mettagrid.test_support.mapgen.render_scene(common_params.model_copy(update={"seed": 77}), (16, 16))

    assert np.array_equal(scene1v1.grid, scene1v2.grid)
    assert not np.array_equal(scene1v1.grid, scene2.grid)
