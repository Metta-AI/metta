import numpy as np
from pytest import fixture

from metta.map.scenes.auto import Auto
from tests.map.scenes.utils import assert_connected, render_scene


@fixture
def common_params():
    return {
        "num_agents": 4,
        "objects": {"altar": 0.02},
        "room_objects": {"altar": ["uniform", 0.0005, 0.01]},
        "room_symmetry": {"horizontal": 1, "vertical": 1, "x4": 1, "none": 1},
        "layout": {"grid": 1, "bsp": 1},
        "grid": {"rows": 3, "columns": 3},
        "bsp": {"area_count": 3},
        "content": [
            {
                "scene": {
                    "type": "metta.map.scenes.maze.Maze",
                    "params": {
                        "room_size": ["uniform", 1, 2],
                        "wall_size": ["uniform", 1, 2],
                    },
                },
                "weight": 3,
            },
        ],
    }


def test_basic(common_params):
    scene = render_scene(Auto, common_params, (10, 10))

    assert_connected(scene.grid)


def test_seed(common_params):
    scene1v1 = render_scene(Auto, common_params, (16, 16), seed=42)
    scene1v2 = render_scene(Auto, common_params, (16, 16), seed=42)
    scene2 = render_scene(Auto, common_params, (16, 16), seed=77)

    assert np.array_equal(scene1v1.grid, scene1v2.grid)
    assert not np.array_equal(scene1v1.grid, scene2.grid)
