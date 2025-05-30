import pytest

from metta.map.scenes.inline_ascii import InlineAscii
from tests.map.scenes.utils import assert_grid, render_node


def test_basic():
    node = render_node(
        InlineAscii,
        {
            "data": """
W A W
W   W
"""
        },
        (3, 7),
    )

    assert_grid(
        node,
        """
|# A #  |
|#   #  |
|       |
""",
    )


def test_row_column():
    node = render_node(
        InlineAscii,
        {
            "data": "W A W",
            "row": 1,
            "column": 2,
        },
        (3, 7),
    )

    assert_grid(
        node,
        """
|       |
|  # A #|
|       |
""",
    )


def test_overflow():
    with pytest.raises(ValueError):
        render_node(
            InlineAscii,
            {
                "data": "WWWW",
                "row": 1,
                "column": 2,
            },
            (1, 3),
        )
