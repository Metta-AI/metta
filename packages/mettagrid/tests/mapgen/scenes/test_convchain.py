import mettagrid.mapgen.scenes.convchain
import mettagrid.test_support.mapgen


def test_basic():
    scene = mettagrid.test_support.mapgen.render_scene(
        mettagrid.mapgen.scenes.convchain.ConvChain.Config(
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
