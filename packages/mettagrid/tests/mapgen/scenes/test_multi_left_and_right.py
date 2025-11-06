import mettagrid.mapgen.scenes.multi_left_and_right
import mettagrid.test_support.mapgen


def test_basic():
    scene = mettagrid.test_support.mapgen.render_scene(
        mettagrid.mapgen.scenes.multi_left_and_right.MultiLeftAndRight.Config(
            rows=3, columns=2, altar_ratio=0.75, total_altars=4
        ),
        shape=(20, 20),
    )

    assert (scene.grid == "wall").sum() > 0
