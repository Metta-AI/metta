from metta.map.scenes.inline_ascii import InlineAscii
from metta.map.scenes.nop import Nop
from metta.map.scenes.remove_agents import RemoveAgents
from tests.map.scenes.utils import assert_grid, scene_to_node


def test_basic():
    scene = Nop(
        children=[
            {
                "scene": InlineAscii("WWW\n" + "AA \n" + "WWW\n"),
                "where": "full",
            },
            {
                "scene": RemoveAgents(),
                "where": "full",
            },
        ]
    )
    node = scene_to_node(scene, (3, 3))

    assert_grid(
        node,
        """
            |###|
            |   |
            |###|
        """,
    )
