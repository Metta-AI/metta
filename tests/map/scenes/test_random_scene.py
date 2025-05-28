from metta.map.scenes.inline_ascii import InlineAscii
from metta.map.scenes.random_scene import RandomScene
from tests.map.scenes.utils import scene_to_node


def test_objects():
    w_count = 0
    a_count = 0

    # 1 / 2^30 chance of failure
    for _ in range(30):
        scene = RandomScene(
            candidates=[
                {"scene": InlineAscii("W"), "weight": 1},
                {"scene": InlineAscii("a"), "weight": 1},
            ]
        )
        node = scene_to_node(scene, (1, 1))
        w_count += (node.grid == "wall").sum()
        a_count += (node.grid == "altar").sum()

    assert w_count > 0
    assert a_count > 0
