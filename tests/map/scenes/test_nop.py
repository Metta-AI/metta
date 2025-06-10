from metta.map.scenes.nop import Nop
from tests.map.scenes.utils import render_scene


def test_basic():
    scene = render_scene(Nop, {}, (3, 3))

    assert (scene.grid == "empty").sum() == 9
