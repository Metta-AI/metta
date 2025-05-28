from metta.map.scenes.ascii import Ascii
from tests.map.scenes.utils import assert_grid, scene_to_node


def test_basic():
    scene = Ascii(uri="tests/map/scenes/fixtures/test.map")
    node = scene_to_node(scene, (4, 4))

    assert_grid(
        node,
        """
|####|
|#a #|
|## #|
|####|
        """,
    )
