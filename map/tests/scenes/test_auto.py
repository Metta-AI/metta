from metta.map.scenes.auto import Auto
from tests.map.scenes.utils import assert_connected, render_scene


def test_basic():
    params = {
        "num_agents": 1,
        "objects": {"altar": 2},
        "room_objects": {"altar": ["uniform", 0.0005, 0.01]},
        "room_symmetry": {"horizontal": 1, "vertical": 1, "x4": 1, "none": 1},
        "layout": {"grid": 1, "bsp": 1},
        "grid": {"rows": 3, "columns": 3},
        "bsp": {"area_count": 3},
        "content": [
            {
                "scene": {
                    "type": "metta.map.scenes.maze.MazeKruskal",
                    "params": {
                        "room_size": ["uniform", 1, 3],
                        "wall_size": ["uniform", 1, 3],
                    },
                },
                "weight": 3,
            },
        ],
    }
    scene = render_scene(Auto, params, (10, 10))

    assert_connected(scene.grid)
