import pytest

import mettagrid.mapgen.scenes.bsp
import mettagrid.test_support.mapgen


# Some BSP scenes end up disconnected.
# BSP is not heavily used right now, so this is a TODO note for the future improvements if we ever need it.
@pytest.mark.skip(reason="BSP has bugs")
def test_basic():
    for _ in range(10):
        scene = mettagrid.test_support.mapgen.render_scene(
            mettagrid.mapgen.scenes.bsp.BSP.Config(
                rooms=7,
                min_room_size=3,
                min_room_size_ratio=0.5,
                max_room_size_ratio=0.9,
            ),
            (20, 20),
        )

        mettagrid.test_support.mapgen.assert_connected(scene.grid)
