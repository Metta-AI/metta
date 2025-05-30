from metta.map.scenes.convchain import ConvChain
from tests.map.scenes.utils import scene_to_node


def test_basic():
    scene = ConvChain(
        pattern="""
|##  #|
|#    |
|#####|
""",
        pattern_size=3,
        iterations=10,
        temperature=1,
    )
    node = scene_to_node(scene, (20, 20))

    assert (node.grid == "wall").sum() > 0
    assert (node.grid == "empty").sum() > 0
