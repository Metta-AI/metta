from metta.map.scenes.wfc import WFC
from tests.map.scenes.utils import scene_to_node


def test_basic():
    scene = WFC(
        pattern="""
| #   |
|###  |
|###  |
""",
    )
    node = scene_to_node(scene, (20, 20))

    assert (node.grid == "wall").sum() > 0
    assert (node.grid == "empty").sum() > 0
