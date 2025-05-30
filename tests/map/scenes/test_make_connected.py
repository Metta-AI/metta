from metta.map.scenes.make_connected import MakeConnected
from metta.map.scenes.room_grid import RoomGrid
from tests.map.scenes.utils import assert_connected, render_node


def test_connect_room_grid():
    node = render_node(
        RoomGrid,
        params=dict(
            rows=2,
            columns=3,
        ),
        shape=(20, 20),
        children=[{"scene": lambda grid: MakeConnected(grid=grid, params={}), "where": "full"}],
    )

    assert_connected(node.grid)
