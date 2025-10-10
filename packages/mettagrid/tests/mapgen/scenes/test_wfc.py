from mettagrid.mapgen.scenes.wfc import WFC
from mettagrid.test_support.mapgen import render_scene


def test_basic():
    scene = render_scene(
        WFC.Config(
            pattern="""
                .#...
                ###..
                ###..
            """
        ),
        (20, 20),
    )

    assert (scene.grid == "wall").sum() > 0
    assert (scene.grid == "empty").sum() > 0
