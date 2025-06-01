from metta.map.scenes.nop import Nop
from tests.map.scenes.utils import scene_to_node


def test_basic():
    scene = Nop()
    node = scene_to_node(scene, (3, 3))

    assert (node.grid == "empty").sum() == 9
