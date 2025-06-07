from metta.map.scenes.ascii import Ascii
from tests.map.scenes.utils import assert_grid, render_scene


def test_basic():
    scene = render_scene(Ascii, {"uri": "tests/map/scenes/fixtures/test.map"}, (4, 4))

    assert_grid(
        scene,
        """
####
#_.#
##.#
####
        """,
    )
