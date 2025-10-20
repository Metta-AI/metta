from mettagrid.mapgen.scenes.random_dcss_scene import RandomDcssScene
from mettagrid.test_support.mapgen import render_scene


def test_basic():
    scene = render_scene(
        RandomDcssScene.Config(wfc=True, dcss=True),
        (10, 10),
    )

    assert scene.grid.shape == (10, 10)
