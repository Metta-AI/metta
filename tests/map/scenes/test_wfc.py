from metta.map.scenes.wfc import WFC
from tests.map.scenes.utils import render_node


def test_basic():
    node = render_node(
        WFC,
        dict(
            pattern="""
| #   |
|###  |
|###  |
""",
        ),
        (20, 20),
    )

    assert (node.grid == "wall").sum() > 0
    assert (node.grid == "empty").sum() > 0
