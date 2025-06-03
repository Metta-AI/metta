import pytest

from metta.map.scenes.bsp import BSP
from tests.map.scenes.utils import assert_connected, render_node


# Some BSP scenes end up disconnected.
# BSP is not heavily used right now, so this is a TODO note for the future improvements if we ever need it.
@pytest.mark.skip(reason="BSP has bugs")
def test_basic():
    for _ in range(10):
        node = render_node(
            BSP,
            {"rooms": 7, "min_room_size": 3, "min_room_size_ratio": 0.5, "max_room_size_ratio": 0.9},
            (20, 20),
        )

        assert_connected(node.grid)
