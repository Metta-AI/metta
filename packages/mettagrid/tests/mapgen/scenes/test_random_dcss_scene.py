import mettagrid.mapgen.scenes.random_dcss_scene
import mettagrid.test_support.mapgen


def test_basic():
    scene = mettagrid.test_support.mapgen.render_scene(
        mettagrid.mapgen.scenes.random_dcss_scene.RandomDcssScene.Config(wfc=True, dcss=True),
        (10, 10),
    )

    assert scene.grid.shape == (10, 10)
