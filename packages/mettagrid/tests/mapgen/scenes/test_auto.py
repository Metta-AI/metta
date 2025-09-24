import numpy as np
from pytest import fixture

from mettagrid.mapgen.random.float import FloatConstantDistribution, FloatUniformDistribution
from mettagrid.mapgen.random.int import IntConstantDistribution, IntUniformDistribution
from mettagrid.mapgen.scenes.auto import (
    Auto,
    AutoParams,
    AutoParamsBSP,
    AutoParamsGrid,
    AutoParamsLayout,
    AutoParamsRoomSymmetry,
)
from mettagrid.mapgen.scenes.maze import Maze
from mettagrid.mapgen.scenes.random_scene import RandomSceneCandidate
from mettagrid.test_support.mapgen import assert_connected, render_scene


@fixture
def common_params() -> AutoParams:
    return AutoParams(
        num_agents=4,
        objects={"altar": FloatConstantDistribution(value=0.02)},
        room_objects={"altar": FloatUniformDistribution(low=0.0005, high=0.01)},
        room_symmetry=AutoParamsRoomSymmetry(horizontal=1, vertical=1, x4=1, none=1),
        layout=AutoParamsLayout(grid=1, bsp=1),
        grid=AutoParamsGrid(rows=IntConstantDistribution(value=3), columns=IntConstantDistribution(value=3)),
        bsp=AutoParamsBSP(area_count=IntConstantDistribution(value=3)),
        content=[
            RandomSceneCandidate(
                scene=Maze.factory(
                    Maze.Params(
                        room_size=IntUniformDistribution(low=1, high=2),
                        wall_size=IntUniformDistribution(low=1, high=2),
                    )
                ),
                weight=3,
            ),
        ],
    )


def test_basic(common_params):
    scene = render_scene(Auto.factory(common_params), (10, 10))

    assert_connected(scene.grid)


def test_seed(common_params):
    scene1v1 = render_scene(Auto.factory(common_params, seed=42), (16, 16))
    scene1v2 = render_scene(Auto.factory(common_params, seed=42), (16, 16))
    scene2 = render_scene(Auto.factory(common_params, seed=77), (16, 16))

    assert np.array_equal(scene1v1.grid, scene1v2.grid)
    assert not np.array_equal(scene1v1.grid, scene2.grid)
