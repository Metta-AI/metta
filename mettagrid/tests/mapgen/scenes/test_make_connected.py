from metta.mettagrid.mapgen.scene import ChildrenAction
from metta.mettagrid.mapgen.scenes.make_connected import MakeConnected
from metta.mettagrid.mapgen.scenes.room_grid import RoomGrid
from tests.mapgen.scenes.utils import assert_connected, render_scene


def test_connect_room_grid():
    scene = render_scene(
        RoomGrid.factory(
            RoomGrid.Params(
                rows=2,
                columns=3,
            ),
            children_actions=[ChildrenAction(scene=MakeConnected.factory(), where="full")],
        ),
        shape=(20, 20),
    )

    assert_connected(scene.grid)
