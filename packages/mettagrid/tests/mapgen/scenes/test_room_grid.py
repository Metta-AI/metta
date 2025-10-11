import numpy as np

from mettagrid.mapgen.area import AreaQuery
from mettagrid.mapgen.scenes.room_grid import RoomGrid
from mettagrid.test_support.mapgen import render_scene

from .test_utils import assert_grid


def test_exact():
    # Test creating a 2x3 grid of rooms
    scene = render_scene(
        RoomGrid.Config(rows=2, columns=3, border_width=1, border_object="wall"),
        (10, 10),
    )

    assert_grid(
        scene,
        """
..#..#..##
..#..#..##
..#..#..##
..#..#..##
##########
..#..#..##
..#..#..##
..#..#..##
..#..#..##
##########
""",
    )


def test_with_rows_columns():
    # Test creating a 2x3 grid of rooms
    scene = render_scene(
        RoomGrid.Config(rows=2, columns=3, border_width=1, border_object="wall"),
        (10, 10),
    )

    # Verify the grid structure
    # Should have walls at inner borders
    assert np.array_equal(scene.grid[4, :], ["wall"] * 10)  # Horizontal border
    assert np.array_equal(scene.grid[:, 2], ["wall"] * 10)  # Vertical border
    areas = scene.select_areas(AreaQuery())
    # Verify room areas are created. The 4x2 shape is due to the border width.
    assert len(areas) == 6
    assert all(area.grid.shape == (4, 2) for area in areas)
    # Verify room positions
    room_positions = [(area.grid[0, 0], area.grid[-1, -1]) for area in areas]
    assert all(pos[0] == "empty" and pos[1] == "empty" for pos in room_positions)


def test_with_layout():
    # Test creating rooms with a specific layout and tags
    layout = [["room1", "room2"], ["room3", "room4"]]

    scene = render_scene(
        RoomGrid.Config(layout=layout, border_width=1, border_object="wall"),
        (10, 10),
    )

    areas = scene.select_areas(AreaQuery())
    # Verify room areas are created with correct tags
    assert len(areas) == 4
    assert all(area.grid.shape == (4, 4) for area in areas)
    # Verifying that the tagged areas are where we expect them to be is a pain,
    # so for now just verify that the tags are what we expect.
    room_tags = [area.tags[0] for area in areas]
    assert set(room_tags) == {"room1", "room2", "room3", "room4"}
