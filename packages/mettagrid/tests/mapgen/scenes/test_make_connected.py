import mettagrid.mapgen.scene
import mettagrid.mapgen.scenes.make_connected
import mettagrid.mapgen.scenes.room_grid
import mettagrid.test_support.mapgen


def test_connect_room_grid():
    scene = mettagrid.test_support.mapgen.render_scene(
        mettagrid.mapgen.scenes.room_grid.RoomGrid.Config(
            rows=2,
            columns=3,
            children=[
                mettagrid.mapgen.scene.ChildrenAction(
                    scene=mettagrid.mapgen.scenes.make_connected.MakeConnected.Config(), where="full"
                )
            ],
        ),
        shape=(20, 20),
    )

    mettagrid.test_support.mapgen.assert_connected(scene.grid)
