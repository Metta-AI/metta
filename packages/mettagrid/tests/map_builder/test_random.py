import numpy as np

from mettagrid.map_builder.map_builder import GameMap
from mettagrid.map_builder.random_map import RandomMapBuilder


class TestRandomMapBuilderConfig:
    def test_create(self):
        objects = {"wall": 5, "empty": 10}
        config = RandomMapBuilder.Config(width=5, height=5, objects=objects, agents=1, seed=42)
        builder = config.create()
        assert isinstance(builder, RandomMapBuilder)

    def test_config_defaults(self):
        objects = {"wall": 5}
        config = RandomMapBuilder.Config(width=5, height=5, objects=objects)
        assert config.agents == 0
        assert config.seed is None
        assert config.border_width == 0
        assert config.border_object == "wall"


class TestRandomMapBuilder:
    def test_build_deterministic_with_seed(self):
        objects = {"wall": 3, "assembler": 2}
        config = RandomMapBuilder.Config(width=4, height=4, objects=objects, agents=1, seed=42)
        builder = config.create()
        map1 = builder.build()

        # Build again with same seed
        builder2 = RandomMapBuilder(config)
        map2 = builder2.build()

        assert np.array_equal(map1.grid, map2.grid)

    def test_build_different_seeds_different_results(self):
        objects = {"wall": 3, "assembler": 2}
        config1 = RandomMapBuilder.Config(width=4, height=4, objects=objects, agents=1, seed=42)
        config2 = RandomMapBuilder.Config(width=4, height=4, objects=objects, agents=1, seed=123)

        builder1 = config1.create()
        builder2 = config2.create()
        map1 = builder1.build()
        map2 = builder2.build()

        # Should be different (very unlikely to be identical by chance)
        assert not np.array_equal(map1.grid, map2.grid)

    def test_build_correct_object_counts(self):
        objects = {"wall": 3, "assembler": 2}
        config = RandomMapBuilder.Config(width=5, height=3, objects=objects, agents=1, seed=42)
        builder = config.create()
        game_map = builder.build()

        # Count objects in the map
        unique, counts = np.unique(game_map.grid, return_counts=True)
        count_dict = dict(zip(unique, counts, strict=False))

        assert count_dict.get("wall", 0) == 3
        assert count_dict.get("assembler", 0) == 2
        assert count_dict.get("agent.agent", 0) == 1
        # Remaining should be empty
        total_objects = 3 + 2 + 1
        total_cells = 5 * 3
        assert count_dict.get("empty", 0) == total_cells - total_objects

    def test_build_with_integer_agents(self):
        objects = {"wall": 2}
        config = RandomMapBuilder.Config(width=3, height=3, objects=objects, agents=2, seed=42)
        builder = config.create()
        game_map = builder.build()

        unique, counts = np.unique(game_map.grid, return_counts=True)
        count_dict = dict(zip(unique, counts, strict=False))

        assert count_dict.get("agent.agent", 0) == 2

    def test_build_with_dictconfig_agents(self):
        objects = {"wall": 2}
        agents = {"agent": 1, "prey": 2}
        config = RandomMapBuilder.Config(width=4, height=3, objects=objects, agents=agents, seed=42)
        builder = config.create()
        game_map = builder.build()

        unique, counts = np.unique(game_map.grid, return_counts=True)
        count_dict = dict(zip(unique, counts, strict=False))

        assert count_dict.get("agent.agent", 0) == 1
        assert count_dict.get("agent.prey", 0) == 2

    def test_build_too_many_objects_halving(self):
        # Create scenario where objects exceed 2/3 of area
        objects = {"wall": 10, "assembler": 10}  # 20 objects
        config = RandomMapBuilder.Config(
            width=5,
            height=5,
            objects=objects,
            agents=0,
            seed=42,  # 25 total cells, 2/3 ≈ 16.67
        )
        builder = config.create()
        game_map = builder.build()

        # Should have halved the object counts
        unique, counts = np.unique(game_map.grid, return_counts=True)
        count_dict = dict(zip(unique, counts, strict=False))

        # Check that total objects don't exceed area
        total_non_empty = sum(count for obj, count in count_dict.items() if obj != "empty")
        assert total_non_empty <= 25

    def test_build_empty_map(self):
        objects = {}
        config = RandomMapBuilder.Config(width=3, height=2, objects=objects, agents=0, seed=42)
        builder = config.create()
        game_map = builder.build()

        # Should be all empty
        assert np.all(game_map.grid == "empty")
        assert game_map.grid.shape == (2, 3)

    def test_build_single_cell(self):
        objects = {"wall": 1}
        config = RandomMapBuilder.Config(width=1, height=1, objects=objects, agents=0, seed=42)
        builder = config.create()
        game_map = builder.build()

        assert game_map.grid.shape == (1, 1)
        assert game_map.grid[0, 0] == "wall"

    def test_build_map_shape(self):
        objects = {"wall": 1}
        config = RandomMapBuilder.Config(width=7, height=4, objects=objects, agents=0, seed=42)
        builder = config.create()
        game_map = builder.build()

        assert game_map.grid.shape == (4, 7)  # height x width

    def test_build_no_agents_int(self):
        objects = {"wall": 2}
        config = RandomMapBuilder.Config(width=3, height=3, objects=objects, agents=0, seed=42)
        builder = config.create()
        game_map = builder.build()

        unique, counts = np.unique(game_map.grid, return_counts=True)
        count_dict = dict(zip(unique, counts, strict=False))

        assert "agent.agent" not in count_dict or count_dict["agent.agent"] == 0

    def test_build_no_agents_empty_dict(self):
        objects = {"wall": 2}
        agents = {}
        config = RandomMapBuilder.Config(width=3, height=3, objects=objects, agents=agents, seed=42)
        builder = config.create()
        game_map = builder.build()

        unique, counts = np.unique(game_map.grid, return_counts=True)
        count_dict = dict(zip(unique, counts, strict=False))

        # No agent types should be present
        agent_types = [key for key in count_dict.keys() if key.startswith("agent.")]
        assert len(agent_types) == 0

    def test_build_returns_numpy_array(self):
        objects = {"wall": 1}
        config = RandomMapBuilder.Config(width=2, height=2, objects=objects, agents=0, seed=42)
        builder = config.create()
        result = builder.build()

        # Should return a GameMap object
        assert isinstance(result, GameMap)
        assert result.grid.dtype.kind == "U"  # Unicode string type

    def test_build_large_map_performance(self):
        """Test that large maps can be built without performance issues"""
        objects = {"wall": 50, "assembler": 20}
        config = RandomMapBuilder.Config(width=50, height=50, objects=objects, agents=10, seed=42)
        builder = config.create()

        # This should complete without issues
        game_map = builder.build()
        assert game_map.grid.shape == (50, 50)

    def test_multiple_agent_types(self):
        objects = {"wall": 2}
        agents = {"agent": 1, "prey": 1, "predator": 1}
        config = RandomMapBuilder.Config(width=4, height=4, objects=objects, agents=agents, seed=42)
        builder = config.create()
        game_map = builder.build()

        unique, counts = np.unique(game_map.grid, return_counts=True)
        count_dict = dict(zip(unique, counts, strict=False))

        assert count_dict.get("agent.agent", 0) == 1
        assert count_dict.get("agent.prey", 0) == 1
        assert count_dict.get("agent.predator", 0) == 1
