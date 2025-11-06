import mettagrid.mapgen.scenes.nop
import mettagrid.test_support.mapgen


def test_basic():
    scene = mettagrid.test_support.mapgen.render_scene(mettagrid.mapgen.scenes.nop.Nop.Config(), (3, 3))

    assert (scene.grid == "empty").sum() == 9
