import pytest

from mettagrid.mapgen.area import AreaQuery, AreaWhere
from mettagrid.mapgen.scenes.layout import Layout, LayoutArea
from mettagrid.test_support.mapgen import render_scene

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------


def _center(xy: int, size: int) -> int:
    """Return the coordinate of the top-left corner that centers an object."""
    return (xy - size) // 2


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_single_centered_area():
    """A single area should be centered and accessible by its tag."""
    grid_shape = (7, 7)  # height, width
    area_w, area_h = 3, 3

    tag = "zone"
    scene = render_scene(
        Layout.Config(
            areas=[
                LayoutArea(width=area_w, height=area_h, tag=tag),
            ]
        ),
        shape=grid_shape,
    )

    areas = scene.select_areas(AreaQuery(where=AreaWhere(tags=[tag])))
    assert len(areas) == 1

    expected_x = _center(grid_shape[1], area_w)
    expected_y = _center(grid_shape[0], area_h)
    area = areas[0]
    assert (area.x, area.y, area.width, area.height) == (
        expected_x,
        expected_y,
        area_w,
        area_h,
    )


def test_area_too_large_raises():
    """An area larger than the grid should raise a ValueError."""
    with pytest.raises(ValueError):
        render_scene(
            Layout.Config(
                areas=[
                    LayoutArea(width=6, height=6, tag="oversized"),
                ]
            ),
            shape=(5, 5),
        )


def test_multiple_areas():
    """Multiple areas should all be created and retrievable by their tags."""
    grid_shape = (5, 5)
    tags = ["a", "b", "c"]
    scene = render_scene(
        Layout.Config(areas=[LayoutArea(width=1, height=1, tag=t) for t in tags]),
        shape=grid_shape,
    )

    for tag in tags:
        areas = scene.select_areas(AreaQuery(where=AreaWhere(tags=[tag])))
        assert len(areas) == 1, f"Area with tag '{tag}' not found"
        area = areas[0]
        assert (area.x, area.y) == (_center(grid_shape[1], 1), _center(grid_shape[0], 1))
