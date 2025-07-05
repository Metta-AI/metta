from metta.map.scenes.inline_ascii import InlineAscii
from metta.map.scenes.random_scene import RandomScene
from tests.map.scenes.utils import render_scene


def test_objects():
    w_count = 0
    a_count = 0

    # 1 / 2^30 chance of failure
    for _ in range(30):
        scene = render_scene(
            RandomScene,
            dict(
                candidates=[
                    {"scene": InlineAscii.factory(params={"data": "#"}), "weight": 1},
                    {"scene": InlineAscii.factory(params={"data": "_"}), "weight": 1},
                ]
            ),
            (1, 1),
        )
        w_count += (scene.grid == "wall").sum()
        a_count += (scene.grid == "altar").sum()

    assert w_count > 0
    assert a_count > 0
