import mettagrid.mapgen.scenes.wfc
import mettagrid.test_support.mapgen


def test_basic():
    scene = mettagrid.test_support.mapgen.render_scene(
        mettagrid.mapgen.scenes.wfc.WFC.Config(
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
