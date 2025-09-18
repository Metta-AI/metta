import pytest

from mettagrid.mapgen.scenes.inline_ascii import InlineAscii
from mettagrid.test_support.mapgen import assert_grid, render_scene


def test_basic():
    scene = render_scene(
        InlineAscii.factory(
            InlineAscii.Params(
                data="""
#.@.#
#...#
"""
            ),
        ),
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
        InlineAscii.factory(
            InlineAscii.Params(
                data="#.@.#",
                row=1,
                column=2,
            )
        ),
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
            InlineAscii.factory(
                InlineAscii.Params(
                    data="####",
                    row=1,
                    column=2,
                )
            ),
            (1, 3),
        )
