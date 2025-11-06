import pytest

import mettagrid.mapgen.scenes.inline_ascii
import mettagrid.test_support.mapgen
import packages.mettagrid.tests.mapgen.scenes.test_utils


def test_basic():
    scene = mettagrid.test_support.mapgen.render_scene(
        mettagrid.mapgen.scenes.inline_ascii.InlineAscii.Config(
            data="""
#.@.#
#...#
""",
        ),
        (3, 7),
    )

    packages.mettagrid.tests.mapgen.scenes.test_utils.assert_grid(
        scene,
        """
#.@.#..
#...#..
.......
""",
    )


def test_row_column():
    scene = mettagrid.test_support.mapgen.render_scene(
        mettagrid.mapgen.scenes.inline_ascii.InlineAscii.Config(
            data="#.@.#",
            row=1,
            column=2,
        ),
        (3, 7),
    )

    packages.mettagrid.tests.mapgen.scenes.test_utils.assert_grid(
        scene,
        """
.......
..#.@.#
.......
""",
    )


def test_overflow():
    with pytest.raises(ValueError):
        mettagrid.test_support.mapgen.render_scene(
            mettagrid.mapgen.scenes.inline_ascii.InlineAscii.Config(
                data="####",
                row=1,
                column=2,
            ),
            (1, 3),
        )
