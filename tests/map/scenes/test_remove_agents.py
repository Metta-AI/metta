from metta.map.scenes.inline_ascii import InlineAscii
from metta.map.scenes.nop import Nop
from metta.map.scenes.remove_agents import RemoveAgents
from tests.map.scenes.utils import assert_grid, render_node


def test_basic():
    node = render_node(
        Nop,
        {},
        (3, 3),
        children=[
            {
                "scene": lambda grid: InlineAscii(grid=grid, params={"data": "WWW\n" + "AA \n" + "WWW\n"}),
                "where": "full",
            },
            {
                "scene": lambda grid: RemoveAgents(grid=grid),
                "where": "full",
            },
        ],
    )

    assert_grid(
        node,
        """
            |###|
            |   |
            |###|
        """,
    )
