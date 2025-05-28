import pytest

from metta.map.scenes.inline_ascii import InlineAscii
from tests.map.scenes.utils import check_grid, scene_to_node


def test_basic():
    scene = InlineAscii(
        data="""
W A W
W   W
"""
    )
    node = scene_to_node(scene, (3, 7))

    check_grid(
        node,
        """
|# A #  |
|#   #  |
|       |
""",
    )


def test_row_column():
    scene = InlineAscii(
        data="W A W",
        row=1,
        column=2,
    )
    node = scene_to_node(scene, (3, 7))

    check_grid(
        node,
        """
|       |
|  # A #|
|       |
""",
    )


def test_overflow():
    scene = InlineAscii(
        data="WWWW",
        row=1,
        column=2,
    )
    with pytest.raises(ValueError):
        scene_to_node(scene, (1, 3))
