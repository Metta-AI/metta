from metta.map.scenes.wfc import WFC
from tests.map.scenes.utils import render_scene


def test_basic():
    scene = render_scene(
        WFC,
        dict(
            pattern="""
        .#...
        ###..
        ###..
            """,
        ),
        (20, 20),
    )

    assert (scene.grid == "wall").sum() > 0
    assert (scene.grid == "empty").sum() > 0
