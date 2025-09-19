from mettagrid.mapgen.scenes.multi_left_and_right import MultiLeftAndRight
from mettagrid.test_support.mapgen import render_scene


def test_basic():
    scene = render_scene(
        MultiLeftAndRight.factory(MultiLeftAndRight.Params(rows=3, columns=2, altar_ratio=0.75, total_altars=4)),
        shape=(20, 20),
    )

    assert (scene.grid == "wall").sum() > 0
