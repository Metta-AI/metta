from metta.map.scenes.inline_ascii import InlineAscii
from metta.map.scenes.nop import Nop
from metta.map.scenes.remove_agents import RemoveAgents
from metta.map.types import ChildrenAction
from tests.map.scenes.utils import assert_grid, render_scene


def test_basic():
    scene = render_scene(
        Nop,
        {},
        (3, 3),
        children=[
            ChildrenAction(
                scene=lambda area: InlineAscii(
                    area=area,
                    params={
                        "data": """
                    ###
                    @@.
                    ###
                        """
                    },
                ),
                where="full",
            ),
            ChildrenAction(
                scene=lambda area: RemoveAgents(area=area),
                where="full",
            ),
        ],
    )

    assert_grid(
        scene,
        """
            ###
            ...
            ###
        """,
    )
