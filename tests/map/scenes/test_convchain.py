from metta.map.scenes.convchain import ConvChain
from tests.map.scenes.utils import render_scene


def test_basic():
    scene = render_scene(
        ConvChain,
        dict(
            pattern="""
##..#
#....
#####
""",
            pattern_size=3,
            iterations=10,
            temperature=1,
        ),
        (20, 20),
    )

    assert (scene.grid == "wall").sum() > 0
    assert (scene.grid == "empty").sum() > 0
