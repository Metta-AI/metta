import numpy as np
import pytest

from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.scenes.inline_ascii import InlineAscii
from mettagrid.mapgen.scenes.nop import Nop
from mettagrid.mapgen.scenes.room_grid import RoomGrid
from mettagrid.mapgen.scenes.transplant_scene import TransplantScene
from mettagrid.test_support.mapgen import assert_raw_grid


class TestMapGenSize:
    def test_basic_dimensions(self):
        (width, height, border_width) = (10, 10, 2)

        mg = MapGen.Config(
            instance=Nop.Config(),
            width=width,
            height=height,
            border_width=border_width,
        ).create()
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
        mg = MapGen.Config(
            instance=Nop.Config(),
        ).create()
        with pytest.raises(ValueError, match="width and height must be provided"):
            mg.build()

    def test_intrinsic_size(self):
        mg = MapGen.Config(
            instance=InlineAscii.Config(data="@"),
            border_width=2,
        ).create()
        level = mg.build()
        assert level.grid.shape == (5, 5)
        assert level.grid[2, 2] == "agent.agent"

    def test_intrinsic_size_with_explicit_dimensions(self):
        mg = MapGen.Config(
            instance=InlineAscii.Config(data="@"),
            width=10,
            height=10,
            border_width=2,
        ).create()
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
        mg = MapGen.Config(
            instance=InlineAscii.Config(data="@"),
            width=width,
            height=height,
            border_width=border_width,
            instances=instances,
            instance_border_width=instance_bw,
        ).create()
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
        mg = MapGen.Config(
            instance=InlineAscii.Config(
                data="""
                        .@.
                        .@.""",
            ),
            # 10 agents, 2 per instance, so 5 instances
            num_agents=10,
            instance_border_width=1,
            border_width=2,
        ).create()
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


class TestMapGenTeamByInstance:
    def test_set_team_by_instance_multiple_instances(self):
        """Test that agents are assigned to teams based on instance number."""
        from mettagrid.mapgen.scenes.random import Random

        mg = MapGen.Config(
            instance=Random.Config(agents=3),
            width=5,
            height=5,
            instances=4,
            set_team_by_instance=True,
            instance_border_width=1,
            border_width=2,
        ).create()
        level = mg.build()

        # Should have 12 total agents (3 per instance × 4 instances)
        total_agents = np.count_nonzero(np.char.startswith(level.grid, "agent"))
        assert total_agents == 12

        # Check that we have agents from each team
        assert np.count_nonzero(level.grid == "agent.team_0") == 3
        assert np.count_nonzero(level.grid == "agent.team_1") == 3
        assert np.count_nonzero(level.grid == "agent.team_2") == 3
        assert np.count_nonzero(level.grid == "agent.team_3") == 3

        # Should have no generic "agent.agent"
        assert np.count_nonzero(level.grid == "agent.agent") == 0

    def test_set_team_by_instance_false(self):
        """Test that default behavior is preserved when flag is False."""
        from mettagrid.mapgen.scenes.random import Random

        mg = MapGen.Config(
            instance=Random.Config(agents=2),
            width=5,
            height=5,
            instances=3,
            set_team_by_instance=False,  # Explicit False
            instance_border_width=1,
            border_width=2,
        ).create()
        level = mg.build()

        # All agents should be "agent.agent"
        assert np.count_nonzero(level.grid == "agent.agent") == 6
        assert np.count_nonzero(np.char.startswith(level.grid, "agent.team_")) == 0

    def test_set_team_by_instance_single_instance(self):
        """Test that single instance gets team_0 when flag is True."""
        from mettagrid.mapgen.scenes.random import Random

        mg = MapGen.Config(
            instance=Random.Config(agents=5),
            width=10,
            height=10,
            instances=1,
            set_team_by_instance=True,
            border_width=2,
        ).create()
        level = mg.build()

        # All 5 agents should be team_0
        assert np.count_nonzero(level.grid == "agent.team_0") == 5
        assert np.count_nonzero(level.grid == "agent.agent") == 0

    def test_set_team_by_instance_with_dict_agents(self):
        """Test that explicit team names in dict aren't overridden."""
        from mettagrid.mapgen.scenes.random import Random

        mg = MapGen.Config(
            instance=Random.Config(agents={"red": 2, "blue": 2, "green": 1}),
            width=5,
            height=5,
            instances=3,
            set_team_by_instance=True,  # should be ignored!
            instance_border_width=1,
            border_width=2,
        ).create()
        level = mg.build()

        # Dict-based agents should preserve their names
        assert np.count_nonzero(level.grid == "agent.red") == 6  # 2 per instance
        assert np.count_nonzero(level.grid == "agent.blue") == 6  # 2 per instance
        assert np.count_nonzero(level.grid == "agent.green") == 3  # 1 per instance
        assert np.count_nonzero(np.char.startswith(level.grid, "agent.team_")) == 0
