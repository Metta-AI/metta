from metta.map.scenes.multi_left_and_right import MultiLeftAndRight
from tests.map.scenes.utils import render_node


def test_basic():
    node = render_node(
        MultiLeftAndRight,
        params=dict(rows=3, columns=2, altar_ratio=0.75, total_altars=4),
        shape=(20, 20),
    )

    assert (node.grid == "wall").sum() > 0
