import os

from mettagrid.mapgen.scenes.ascii import Ascii
from mettagrid.test_support.mapgen import assert_grid, render_scene


def test_basic():
    uri = f"{os.path.dirname(__file__)}/fixtures/test.map"
    scene = render_scene(Ascii.Config(uri=uri), (4, 4))

    assert_grid(
        scene,
        """
####
#_.#
##.#
####
        """,
    )
