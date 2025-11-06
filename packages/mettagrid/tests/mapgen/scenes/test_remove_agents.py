import mettagrid.mapgen.scene
import mettagrid.mapgen.scenes.inline_ascii
import mettagrid.mapgen.scenes.nop
import mettagrid.mapgen.scenes.remove_agents
import mettagrid.test_support.mapgen
import packages.mettagrid.tests.mapgen.scenes.test_utils


def test_basic():
    scene = mettagrid.test_support.mapgen.render_scene(
        mettagrid.mapgen.scenes.nop.Nop.Config(
            children=[
                mettagrid.mapgen.scene.ChildrenAction(
                    scene=mettagrid.mapgen.scenes.inline_ascii.InlineAscii.Config(
                        data="""
                            ###
                            @@.
                            ###
                        """
                    ),
                    where="full",
                ),
                mettagrid.mapgen.scene.ChildrenAction(
                    scene=mettagrid.mapgen.scenes.remove_agents.RemoveAgents.Config(),
                    where="full",
                ),
            ],
        ),
        (3, 3),
    )

    packages.mettagrid.tests.mapgen.scenes.test_utils.assert_grid(
        scene,
        """
            ###
            ...
            ###
        """,
    )
