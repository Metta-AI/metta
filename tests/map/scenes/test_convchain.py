from metta.map.scenes.convchain import ConvChain
from tests.map.scenes.utils import render_node


def test_basic():
    node = render_node(
        ConvChain,
        dict(
            pattern="""
|##  #|
|#    |
|#####|
""",
            pattern_size=3,
            iterations=10,
            temperature=1,
        ),
        (20, 20),
    )

    assert (node.grid == "wall").sum() > 0
    assert (node.grid == "empty").sum() > 0
