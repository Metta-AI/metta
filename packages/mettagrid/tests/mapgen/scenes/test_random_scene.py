from mettagrid.mapgen.scenes.inline_ascii import InlineAscii
from mettagrid.mapgen.scenes.random_scene import RandomScene, RandomSceneCandidate
from mettagrid.test_support.mapgen import render_scene


def test_objects():
    w_count = 0
    a_count = 0

    # 1 / 2^30 chance of failure
    for _ in range(30):
        scene = render_scene(
            RandomScene.Config(
                candidates=[
                    RandomSceneCandidate(scene=InlineAscii.Config(data="#"), weight=1),
                    RandomSceneCandidate(scene=InlineAscii.Config(data="_"), weight=1),
                ]
            ),
            (1, 1),
        )
        w_count += (scene.grid == "wall").sum()
        a_count += (scene.grid == "assembler").sum()

    assert w_count > 0
    assert a_count > 0
