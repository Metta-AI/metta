from metta.map.scenes.make_connected import MakeConnected
from metta.map.scenes.room_grid import RoomGrid
from tests.map.scenes.utils import is_connected, scene_to_node


def test_connect_room_grid():
    scene = RoomGrid(rows=2, columns=3, children=[{"scene": MakeConnected(), "where": "full"}])
    node = scene_to_node(scene, (20, 20))

    assert is_connected(node.grid)
