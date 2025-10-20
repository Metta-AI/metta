from mettagrid.mapgen.scenes.nop import Nop
from mettagrid.test_support.mapgen import render_scene


def test_basic():
    scene = render_scene(Nop.Config(), (3, 3))

    assert (scene.grid == "empty").sum() == 9
