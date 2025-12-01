from mettagrid.mapgen.scenes.multi_left_and_right import MultiLeftAndRight
from mettagrid.test_support.mapgen import render_scene


def test_basic():
    scene = render_scene(
        MultiLeftAndRight.Config(rows=3, columns=2, assembler_ratio=0.75, total_assemblers=4),
        shape=(20, 20),
    )

    assert (scene.grid == "wall").sum() > 0
