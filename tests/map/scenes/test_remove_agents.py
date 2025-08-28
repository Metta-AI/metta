from metta.map.scene import ChildrenAction
from metta.map.scenes.inline_ascii import InlineAscii
from metta.map.scenes.nop import Nop
from metta.map.scenes.remove_agents import RemoveAgents
from tests.map.scenes.utils import assert_grid, render_scene


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