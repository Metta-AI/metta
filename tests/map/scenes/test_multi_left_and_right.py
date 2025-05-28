from metta.map.scenes.multi_left_and_right import MultiLeftAndRight
from tests.map.scenes.utils import scene_to_node


def test_basic():
    scene = MultiLeftAndRight(rows=3, columns=2)
    node = scene_to_node(scene, (20, 20))

    assert (node.grid == "wall").sum() > 0
