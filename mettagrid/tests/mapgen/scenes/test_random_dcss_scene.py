from metta.mettagrid.mapgen.scenes.random_dcss_scene import RandomDcssScene
from tests.mapgen.scenes.utils import render_scene


def test_basic():
    scene = render_scene(
        RandomDcssScene.factory(RandomDcssScene.Params(wfc=True, dcss=True)),
        (10, 10),
    )

    assert scene.grid.shape == (10, 10)
