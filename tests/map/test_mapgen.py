import numpy as np
import pytest

from metta.common.util.config import Config
from metta.map.mapgen import MapGen
from metta.map.scene import Scene
from metta.map.scenes.room_grid import RoomGrid


# 1. width, height, border_width ------------------------------------------------
@pytest.mark.parametrize(
    "width,height,border_width",
    [(5, 4, 2), (7, 3, 1)],
)
def test_mapgen_basic_dimensions(width, height, border_width):
    mg = MapGen(
        root={"type": "metta.map.scenes.nop.Nop"},
        width=width,
        height=height,
        border_width=border_width,
    )
    level = mg.build()
    expected_shape = (height + 2 * border_width, width + 2 * border_width)
    assert level.grid.shape == expected_shape

    # Verify border is filled with "wall" and interior with "empty"
    bw = border_width
    grid = level.grid
    assert np.all(grid[:bw, :] == "wall")
    assert np.all(grid[-bw:, :] == "wall")
    assert np.all(grid[:, :bw] == "wall")
    assert np.all(grid[:, -bw:] == "wall")
    assert np.all(grid[bw:-bw, bw:-bw] == "empty")


# 2. instances and instance_border_width ---------------------------------------
@pytest.mark.parametrize(
    "instances,instance_bw",
    [
        (4, 1),
        (9, 2),
        (24, 1),  # make sure it's 24 rooms, not 25 (5x5 grid but last room is empty)
    ],
)
def test_mapgen_instances(instances, instance_bw):
    width, height, border_width = 5, 3, 2
    mg = MapGen(
        root={"type": "metta.map.scenes.inline_ascii.InlineAscii", "params": {"data": "@"}},
        width=width,
        height=height,
        border_width=border_width,
        instances=instances,
        instance_border_width=instance_bw,
    )
    level = mg.build()

    rows = int(np.ceil(np.sqrt(instances)))
    cols = int(np.ceil(instances / rows))
    inner_w = width * cols + (cols - 1) * instance_bw
    inner_h = height * rows + (rows - 1) * instance_bw
    expected_shape = (inner_h + 2 * border_width, inner_w + 2 * border_width)
    assert level.grid.shape == expected_shape
    assert isinstance(mg.root_scene, RoomGrid)
    assert np.count_nonzero(np.char.startswith(level.grid, "agent")) == instances


# 3. intrinsic_size interplay ---------------------------------------------------
class DummyParams(Config):
    pass


class DummyIntrinsicallySizedScene(Scene[DummyParams]):
    """Minimal scene with a fixed intrinsic size."""

    @classmethod
    def intrinsic_size(cls, params):
        return 4, 6  # (height, width)

    def render(self):
        pass


DUMMY_SCENE_PATH = f"{DummyIntrinsicallySizedScene.__module__}.{DummyIntrinsicallySizedScene.__qualname__}"


def test_mapgen_intrinsic_size_used_when_dimensions_missing():
    mg = MapGen(root={"type": DUMMY_SCENE_PATH}, border_width=1)
    level = mg.build()
    assert (mg.height, mg.width) == (4, 6)
    assert level.grid.shape == (4 + 2, 6 + 2)


def test_mapgen_intrinsic_size_overrides_when_one_dimension_missing():
    mg = MapGen(root={"type": DUMMY_SCENE_PATH}, width=8, border_width=1)
    level = mg.build()
    # Both dimensions should follow intrinsic_size, ignoring provided width
    assert (mg.height, mg.width) == (4, 6)
    assert level.grid.shape == (4 + 2, 6 + 2)


def test_mapgen_dimensions_preserved_when_provided():
    mg = MapGen(root={"type": DUMMY_SCENE_PATH}, width=7, height=3, border_width=1)
    level = mg.build()
    assert (mg.height, mg.width) == (3, 7)
    assert level.grid.shape == (3 + 2, 7 + 2)


# 4. size-based labels ----------------------------------------------------------
@pytest.mark.parametrize(
    "width,height,expected_label",
    [
        (10, 10, "small"),  # area = 100
        (70, 70, "medium"),  # area = 4 900
        (100, 80, "large"),  # area = 8 000
    ],
)
def test_mapgen_size_labels(width, height, expected_label):
    mg = MapGen(
        root={"type": "metta.map.scenes.nop.Nop"},
        width=width,
        height=height,
    )
    level = mg.build()

    # Exactly one size label should be present and it must match expectation.
    size_labels = {"small", "medium", "large"}
    present = set(level.labels) & size_labels
    assert present == {expected_label}
