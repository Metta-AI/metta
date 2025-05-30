from metta.map.scenes.random_objects import RandomObjects
from tests.map.scenes.utils import render_scene


def test_objects():
    scene = render_scene(
        RandomObjects,
        dict(object_ranges={"mine": ("uniform", 0.2, 0.5)}),
        (10, 10),
    )

    mine_count = (scene.grid == "mine").sum()
    assert 0.2 * 100 <= mine_count <= 0.5 * 100
