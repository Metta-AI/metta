from metta.mettagrid.mapgen.scenes.convchain import ConvChain
from tests.mapgen.scenes.utils import render_scene


def test_basic():
    scene = render_scene(
        ConvChain.factory(
            ConvChain.Params(
                pattern="""
                    ##..#
                    #....
                    #####
                """,
                pattern_size=3,
                iterations=10,
                temperature=1,
            )
        ),
        (20, 20),
    )

    assert (scene.grid == "wall").sum() > 0
    assert (scene.grid == "empty").sum() > 0
