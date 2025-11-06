import mettagrid.mapgen.scenes.inline_ascii
import mettagrid.mapgen.scenes.random_scene
import mettagrid.test_support.mapgen


def test_objects():
    w_count = 0
    a_count = 0

    # 1 / 2^30 chance of failure
    for _ in range(30):
        scene = mettagrid.test_support.mapgen.render_scene(
            mettagrid.mapgen.scenes.random_scene.RandomScene.Config(
                candidates=[
                    mettagrid.mapgen.scenes.random_scene.RandomSceneCandidate(
                        scene=mettagrid.mapgen.scenes.inline_ascii.InlineAscii.Config(data="#"), weight=1
                    ),
                    mettagrid.mapgen.scenes.random_scene.RandomSceneCandidate(
                        scene=mettagrid.mapgen.scenes.inline_ascii.InlineAscii.Config(data="_"), weight=1
                    ),
                ]
            ),
            (1, 1),
        )
        w_count += (scene.grid == "wall").sum()
        a_count += (scene.grid == "altar").sum()

    assert w_count > 0
    assert a_count > 0
