import numpy as np
from pytest import fixture

from mettagrid.mapgen.random.float import FloatConstantDistribution, FloatUniformDistribution
from mettagrid.mapgen.random.int import IntConstantDistribution, IntUniformDistribution
from mettagrid.mapgen.scenes.auto import (
    AutoConfig,
    AutoConfigBSP,
    AutoConfigGrid,
    AutoConfigLayout,
    AutoConfigRoomSymmetry,
)
from mettagrid.mapgen.scenes.maze import Maze
from mettagrid.mapgen.scenes.random_scene import RandomSceneCandidate
from mettagrid.test_support.mapgen import assert_connected, render_scene


@fixture
def common_params() -> AutoConfig:
    return AutoConfig(
        num_agents=4,
        objects={"assembler": FloatConstantDistribution(value=0.02)},
        room_objects={"assembler": FloatUniformDistribution(low=0.0005, high=0.01)},
        room_symmetry=AutoConfigRoomSymmetry(horizontal=1, vertical=1, x4=1, none=1),
        layout=AutoConfigLayout(grid=1, bsp=1),
        grid=AutoConfigGrid(rows=IntConstantDistribution(value=3), columns=IntConstantDistribution(value=3)),
        bsp=AutoConfigBSP(area_count=IntConstantDistribution(value=3)),
        content=[
            RandomSceneCandidate(
                scene=Maze.Config(
                    room_size=IntUniformDistribution(low=1, high=2),
                    wall_size=IntUniformDistribution(low=1, high=2),
                ),
                weight=3,
            ),
        ],
    )


def test_basic(common_params):
    scene = render_scene(common_params, (10, 10))

    assert_connected(scene.grid)


def test_seed(common_params):
    scene1v1 = render_scene(common_params.model_copy(update={"seed": 42}), (16, 16))
    scene1v2 = render_scene(common_params.model_copy(update={"seed": 42}), (16, 16))
    scene2 = render_scene(common_params.model_copy(update={"seed": 77}), (16, 16))

    assert np.array_equal(scene1v1.grid, scene1v2.grid)
    assert not np.array_equal(scene1v1.grid, scene2.grid)
