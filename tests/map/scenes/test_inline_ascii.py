import pytest

from metta.map.scenes.inline_ascii import InlineAscii
from tests.map.scenes.utils import assert_grid, render_scene


def test_basic():
    scene = render_scene(
        InlineAscii,
        {
            "data": """
#.@.#
#...#
"""
        },
        (3, 7),
    )

    assert_grid(
        scene,
        """
#.@.#..
#...#..
.......
""",
    )


def test_row_column():
    scene = render_scene(
        InlineAscii,
        {
            "data": "#.@.#",
            "row": 1,
            "column": 2,
        },
        (3, 7),
    )

    assert_grid(
        scene,
        """
.......
..#.@.#
.......
""",
    )


def test_overflow():
    with pytest.raises(ValueError):
        render_scene(
            InlineAscii,
            {
                "data": "####",
                "row": 1,
                "column": 2,
            },
            (1, 3),
        )
