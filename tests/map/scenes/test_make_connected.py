from metta.map.scenes.make_connected import MakeConnected
from metta.map.scenes.room_grid import RoomGrid
from metta.map.types import ChildrenAction
from tests.map.scenes.utils import assert_connected, render_scene


def test_connect_room_grid():
    scene = render_scene(
        RoomGrid,
        params=dict(
            rows=2,
            columns=3,
        ),
        shape=(20, 20),
        children=[ChildrenAction(scene=MakeConnected.factory(params={}), where="full")],
    )

    assert_connected(scene.grid)
