from metta.map.scenes.bsp import BSP
from tests.map.scenes.utils import is_connected, scene_to_node


def test_basic():
    scene = BSP(rooms=7, min_room_size=3, min_room_size_ratio=0.5, max_room_size_ratio=0.9)
    node = scene_to_node(scene, (20, 20))

    assert is_connected(node.grid)
