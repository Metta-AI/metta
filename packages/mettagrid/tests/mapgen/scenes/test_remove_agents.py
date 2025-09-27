from mettagrid.mapgen.scene import ChildrenAction
from mettagrid.mapgen.scenes.inline_ascii import InlineAscii
from mettagrid.mapgen.scenes.nop import Nop
from mettagrid.mapgen.scenes.remove_agents import RemoveAgents
from mettagrid.test_support.mapgen import assert_grid, render_scene


def test_basic():
    scene = render_scene(
        Nop.Config(
            children=[
                ChildrenAction(
                    scene=InlineAscii.Config(
                        data="""
                            ###
                            @@.
                            ###
                        """
                    ),
                    where="full",
                ),
                ChildrenAction(
                    scene=RemoveAgents.Config(),
                    where="full",
                ),
            ],
        ),
        (3, 3),
    )

    assert_grid(
        scene,
        """
            ###
            ...
            ###
        """,
    )
