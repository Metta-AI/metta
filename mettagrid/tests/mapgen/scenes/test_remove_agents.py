from metta.mettagrid.mapgen.scene import ChildrenAction
from metta.mettagrid.mapgen.scenes.inline_ascii import InlineAscii
from metta.mettagrid.mapgen.scenes.nop import Nop
from metta.mettagrid.mapgen.scenes.remove_agents import RemoveAgents
from tests.mapgen.scenes.utils import assert_grid, render_scene


def test_basic():
    scene = render_scene(
        Nop.factory(
            children_actions=[
                ChildrenAction(
                    scene=InlineAscii.factory(
                        InlineAscii.Params(
                            data="""
                            ###
                            @@.
                            ###
                        """
                        )
                    ),
                    where="full",
                ),
                ChildrenAction(
                    scene=RemoveAgents.factory(),
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
