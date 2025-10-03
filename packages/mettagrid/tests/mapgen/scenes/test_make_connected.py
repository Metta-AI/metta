from mettagrid.mapgen.scene import ChildrenAction
from mettagrid.mapgen.scenes.make_connected import MakeConnected
from mettagrid.mapgen.scenes.room_grid import RoomGrid
from mettagrid.test_support.mapgen import assert_connected, render_scene


def test_connect_room_grid():
    scene = render_scene(
        RoomGrid.Config(
            rows=2,
            columns=3,
            children=[ChildrenAction(scene=MakeConnected.Config(), where="full")],
        ),
        shape=(20, 20),
    )

    assert_connected(scene.grid)
