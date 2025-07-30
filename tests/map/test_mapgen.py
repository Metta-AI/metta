import numpy as np
import pytest

from metta.map.mapgen import MapGen
from metta.map.scenes.inline_ascii import InlineAscii
from metta.map.scenes.room_grid import RoomGrid
from metta.map.scenes.transplant_scene import TransplantScene
from tests.map.scenes.utils import assert_raw_grid


class TestMapGenSize:
    def test_basic_dimensions(self):
        (width, height, border_width) = (10, 10, 2)

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

    def test_dimensions_required(self):
        mg = MapGen(
            root={"type": "metta.map.scenes.nop.Nop"},
        )
        with pytest.raises(ValueError, match="width and height must be provided"):
            mg.build()

    def test_intrinsic_size(self):
        mg = MapGen(
            root={"type": "metta.map.scenes.inline_ascii.InlineAscii", "params": {"data": "@"}},
            border_width=2,
        )
        level = mg.build()
        assert level.grid.shape == (5, 5)
        assert level.grid[2, 2] == "agent.agent"

    def test_intrinsic_size_with_explicit_dimensions(self):
        mg = MapGen(
            root={"type": "metta.map.scenes.inline_ascii.InlineAscii", "params": {"data": "@"}},
            width=10,
            height=10,
            border_width=2,
        )
        level = mg.build()
        assert level.grid.shape == (14, 14)


class TestMapGenInstances:
    @pytest.mark.parametrize(
        "instances,instance_bw",
        [
            (4, 1),
            (9, 2),
            (24, 1),  # make sure it's 24 rooms, not 25 (5x5 grid but last room is empty)
        ],
    )
    def test_instances(self, instances, instance_bw):
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

    def test_num_agents(self):
        mg = MapGen(
            root={
                "type": "metta.map.scenes.inline_ascii.InlineAscii",
                "params": {
                    "data": """
                        .@.
                        .@.
                    """,
                },
            },
            # 10 agents, 2 per instance, so 5 instances
            num_agents=10,
            instance_border_width=1,
            border_width=2,
        )
        level = mg.build()
        assert_raw_grid(
            level.grid,
            """
###########
###########
##.@.#.@.##
##.@.#.@.##
###########
##.@.#.@.##
##.@.#.@.##
###########
##.@.#...##
##.@.#...##
###########
###########
    """,
        )
        assert np.count_nonzero(np.char.startswith(level.grid, "agent")) == 10

        assert isinstance(mg.root_scene, RoomGrid)
        assert isinstance(mg.root_scene.children[0], TransplantScene)

        # first instance is transplanted
        assert isinstance(mg.root_scene.children[0].children[0], InlineAscii)
        assert isinstance(mg.root_scene.children[1], InlineAscii)

        # sanity checks for transplant
        first_child = mg.root_scene.children[0]
        assert first_child.width == 3
        assert first_child.height == 2
        assert first_child.area.x == 2
        assert first_child.area.y == 2


class TestMapGenLabels:
    @pytest.mark.parametrize(
        "width,height,expected_label",
        [
            (10, 10, "small"),  # area = 100
            (70, 70, "medium"),  # area = 4 900
            (100, 80, "large"),  # area = 8 000
        ],
    )
    def test_size_labels(self, width, height, expected_label):
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
