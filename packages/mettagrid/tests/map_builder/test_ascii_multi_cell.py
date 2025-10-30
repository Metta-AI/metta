"""Test ASCII map builder with multi-cell grouping."""

import numpy as np

from mettagrid.config.mettagrid_config import GameConfig, MettaGridConfig, WallConfig
from mettagrid.core import MettaGridCore
from mettagrid.map_builder.ascii import AsciiMapBuilder


def test_ascii_multi_cell_grouping():
    """Test that adjacent walls are grouped into multi-cell objects."""
    map_data = [
        "###",
        "#.#",
        "###",
    ]

    config = AsciiMapBuilder.Config(
        map_data=map_data,
        char_to_name_map={
            "#": "wall",
            ".": "empty",
        },
        auto_group_types=["wall"],
    )

    builder = config.create()
    game_map = builder.build()

    # Check that grid has been modified
    # The top-left wall should remain, others replaced with empty
    assert game_map.grid[0, 0] == "wall"  # primary location
    # Other walls in the component should be empty
    empty_count = np.count_nonzero(game_map.grid == "empty")
    assert empty_count > 1  # At least the extra wall locations + the original empty cell

    # Check multi_cell_groups metadata
    assert len(game_map.multi_cell_groups) == 1  # One connected component
    primary_r, primary_c, layer, extra_locations = game_map.multi_cell_groups[0]
    assert layer == 1  # Walls are on ObjectLayer (1)
    # 8 walls minus 1 primary = 7 extra locations
    assert len(extra_locations) >= 7


def test_ascii_multi_cell_with_env():
    """Test that multi-cell objects work in an actual environment."""
    # Use a simple environment with just walls to test multi-cell grouping
    # Without agents (simplest case)

    mg_config = MettaGridConfig(
        game=GameConfig(
            num_agents=1,  # Must have at least 1 agent
            map_builder=AsciiMapBuilder.Config(
                map_data=[
                    "#####",
                    "#...#",
                    "#.@.#",
                    "#...#",
                    "#####",
                ],
                char_to_name_map={
                    "#": "wall",
                    ".": "empty",
                    "@": "agent.agent",
                },
                auto_group_types=["wall"],
            ),
            objects={
                "wall": WallConfig(
                    type_id=1,
                    name="wall",
                    map_char="#",
                    render_symbol="⬛",
                ),
            },
        )
    )

    env = MettaGridCore(mg_config)
    env.reset()

    # Get grid objects
    objects = env.grid_objects()

    # Count wall objects
    walls = [obj for obj in objects.values() if obj.get("type_name") == "wall"]

    # Should have only 1 wall object (the grouped one)
    assert len(walls) == 1, f"Expected 1 wall object, got {len(walls)}"

    # Check that wall has multiple locations
    wall = walls[0]
    locations = wall.get("locations", [])
    # 5x5 map with 4x4 interior means border has 16 walls (but corners)
    # Actually: 5+5+5+5 - 4 corners = 16 walls
    assert len(locations) == 16, f"Expected 16 locations in border wall, got {len(locations)}"


def test_ascii_no_grouping_by_default():
    """Test that without auto_group_types, objects are not grouped."""
    map_data = [
        "###",
        "#.#",
        "###",
    ]

    config = AsciiMapBuilder.Config(
        map_data=map_data,
        char_to_name_map={
            "#": "wall",
            ".": "empty",
        },
        # No auto_group_types - should default to empty list
    )

    builder = config.create()
    game_map = builder.build()

    # Without grouping, all walls should remain in grid
    wall_count = np.count_nonzero(game_map.grid == "wall")
    assert wall_count == 8

    # No multi-cell groups
    assert len(game_map.multi_cell_groups) == 0


def test_ascii_multiple_components():
    """Test that separate wall groups are handled correctly."""
    map_data = [
        "##.##",
        ".....",
        "##.##",
    ]

    config = AsciiMapBuilder.Config(
        map_data=map_data,
        char_to_name_map={
            "#": "wall",
            ".": "empty",
        },
        auto_group_types=["wall"],
    )

    builder = config.create()
    game_map = builder.build()

    # Should have 4 components (top-left, top-right, bottom-left, bottom-right)
    assert len(game_map.multi_cell_groups) == 4

    # Each component should have 1 extra cell (2 walls - 1 primary)
    for _, _, _, extra_locations in game_map.multi_cell_groups:
        assert len(extra_locations) == 1


def test_ascii_layer_override():
    """Per-type layer overrides should take precedence over heuristics."""
    map_data = [
        "AA",
        "A.",
    ]

    config = AsciiMapBuilder.Config(
        map_data=map_data,
        char_to_name_map={
            "A": "agent.special",
            ".": "empty",
        },
        auto_group_types=["agent.special"],
        auto_group_layer_overrides={"agent.special": 1},
    )

    game_map = config.create().build()

    assert len(game_map.multi_cell_groups) == 1
    _, _, layer, extra_locations = game_map.multi_cell_groups[0]
    assert layer == 1
    assert len(extra_locations) == 2
