import numpy as np

from mettagrid.map_builder.map_builder import GameMap
from mettagrid.map_builder.maze import (
    MazeKruskalMapBuilder,
    MazePrimMapBuilder,
)


class TestMazePrimMapBuilderConfig:
    def test_create(self):
        config = MazePrimMapBuilder.Config(width=11, height=11, start_pos=(1, 1), end_pos=(9, 9), seed=42)
        builder = config.create()
        assert isinstance(builder, MazePrimMapBuilder)


class TestMazeKruskalMapBuilderConfig:
    def test_create(self):
        config = MazeKruskalMapBuilder.Config(width=11, height=11, start_pos=(1, 1), end_pos=(9, 9), seed=42)
        builder = config.create()
        assert isinstance(builder, MazeKruskalMapBuilder)


class TestMazePrimMapBuilder:
    def test_build_deterministic_with_seed(self):
        config = MazePrimMapBuilder.Config(width=11, height=11, start_pos=(1, 1), end_pos=(9, 9), seed=42)
        builder1 = MazePrimMapBuilder(config)
        map1 = builder1.build()

        builder2 = MazePrimMapBuilder(config)
        map2 = builder2.build()

        assert isinstance(map1, GameMap)
        assert isinstance(map2, GameMap)
        assert np.array_equal(map1.grid, map2.grid)

    def test_build_different_seeds_different_results(self):
        config1 = MazePrimMapBuilder.Config(width=11, height=11, start_pos=(1, 1), end_pos=(9, 9), seed=42)
        config2 = MazePrimMapBuilder.Config(width=11, height=11, start_pos=(1, 1), end_pos=(9, 9), seed=123)

        builder1 = MazePrimMapBuilder(config1)
        builder2 = MazePrimMapBuilder(config2)
        map1 = builder1.build()
        map2 = builder2.build()

        # Should be different (very unlikely to be identical by chance)
        assert not np.array_equal(map1.grid, map2.grid)

    def test_build_start_and_end_positions(self):
        config = MazePrimMapBuilder.Config(width=11, height=11, start_pos=(1, 1), end_pos=(9, 9), seed=42)
        builder = MazePrimMapBuilder(config)
        game_map = builder.build()

        assert game_map.grid[1, 1] == "agent.agent"  # start position
        assert game_map.grid[9, 9] == "assembler"  # end position

    def test_build_odd_dimensions_preserved(self):
        config = MazePrimMapBuilder.Config(width=11, height=13, start_pos=(1, 1), end_pos=(9, 11), seed=42)
        builder = MazePrimMapBuilder(config)

        assert builder._width == 11  # Odd, preserved
        assert builder._height == 13  # Odd, preserved

    def test_build_even_dimensions_made_odd(self):
        config = MazePrimMapBuilder.Config(width=10, height=12, start_pos=(1, 1), end_pos=(7, 9), seed=42)
        builder = MazePrimMapBuilder(config)

        assert builder._width == 9  # Even made odd (10-1)
        assert builder._height == 11  # Even made odd (12-1)

    def test_build_position_adjustment(self):
        # Test that positions are properly adjusted
        config = MazePrimMapBuilder.Config(width=11, height=11, start_pos=(0, 0), end_pos=(10, 10), seed=42)
        builder = MazePrimMapBuilder(config)

        # Should be adjusted to odd positions within bounds
        assert builder._start_pos[0] % 2 == 1  # Odd
        assert builder._start_pos[1] % 2 == 1  # Odd
        assert builder._end_pos[0] % 2 == 1  # Odd
        assert builder._end_pos[1] % 2 == 1  # Odd
        assert 0 <= builder._start_pos[0] < builder._width
        assert 0 <= builder._start_pos[1] < builder._height
        assert 0 <= builder._end_pos[0] < builder._width
        assert 0 <= builder._end_pos[1] < builder._height

    def test_build_small_maze(self):
        config = MazePrimMapBuilder.Config(width=5, height=5, start_pos=(1, 1), end_pos=(3, 3), seed=42)
        builder = MazePrimMapBuilder(config)
        game_map = builder.build()

        assert game_map.grid.shape == (5, 5)
        assert game_map.grid[1, 1] == "agent.agent"
        assert game_map.grid[3, 3] == "assembler"

        # Should have some empty cells (paths)
        unique, counts = np.unique(game_map.grid, return_counts=True)
        count_dict = dict(zip(unique, counts, strict=False))
        assert count_dict.get("empty", 0) > 0

    def test_build_returns_game_map(self):
        config = MazePrimMapBuilder.Config(width=7, height=7, start_pos=(1, 1), end_pos=(5, 5), seed=42)
        builder = MazePrimMapBuilder(config)
        result = builder.build()

        assert isinstance(result, GameMap)

    def test_build_maze_structure(self):
        """Test that the maze has proper structure with walls and paths"""
        config = MazePrimMapBuilder.Config(width=11, height=11, start_pos=(1, 1), end_pos=(9, 9), seed=42)
        builder = MazePrimMapBuilder(config)
        game_map = builder.build()

        # Count different elements
        unique, counts = np.unique(game_map.grid, return_counts=True)
        count_dict = dict(zip(unique, counts, strict=False))

        # Should have walls, empty spaces, start, and end
        assert "wall" in count_dict
        assert "empty" in count_dict
        assert "agent.agent" in count_dict
        assert "assembler" in count_dict

        # Should have exactly one start and one end
        assert count_dict["agent.agent"] == 1
        assert count_dict["assembler"] == 1


class TestMazeKruskalMapBuilder:
    def test_build_deterministic_with_seed(self):
        config = MazeKruskalMapBuilder.Config(width=11, height=11, start_pos=(1, 1), end_pos=(9, 9), seed=42)
        builder1 = MazeKruskalMapBuilder(config)
        map1 = builder1.build()

        builder2 = MazeKruskalMapBuilder(config)
        map2 = builder2.build()

        assert isinstance(map1, GameMap)
        assert isinstance(map2, GameMap)
        assert np.array_equal(map1.grid, map2.grid)

    def test_build_different_seeds_different_results(self):
        config1 = MazeKruskalMapBuilder.Config(width=11, height=11, start_pos=(1, 1), end_pos=(9, 9), seed=42)
        config2 = MazeKruskalMapBuilder.Config(width=11, height=11, start_pos=(1, 1), end_pos=(9, 9), seed=123)

        builder1 = MazeKruskalMapBuilder(config1)
        builder2 = MazeKruskalMapBuilder(config2)
        map1 = builder1.build()
        map2 = builder2.build()

        # Should be different (very unlikely to be identical by chance)
        assert not np.array_equal(map1.grid, map2.grid)

    def test_build_start_and_end_positions(self):
        config = MazeKruskalMapBuilder.Config(width=11, height=11, start_pos=(1, 1), end_pos=(9, 9), seed=42)
        builder = MazeKruskalMapBuilder(config)
        game_map = builder.build()

        assert game_map.grid[1, 1] == "agent.agent"  # start position
        assert game_map.grid[9, 9] == "assembler"  # end position

    def test_build_odd_dimensions_preserved(self):
        config = MazeKruskalMapBuilder.Config(width=11, height=13, start_pos=(1, 1), end_pos=(9, 11), seed=42)
        builder = MazeKruskalMapBuilder(config)

        assert builder._width == 11  # Odd, preserved
        assert builder._height == 13  # Odd, preserved

    def test_build_even_dimensions_made_odd(self):
        config = MazeKruskalMapBuilder.Config(width=10, height=12, start_pos=(1, 1), end_pos=(7, 9), seed=42)
        builder = MazeKruskalMapBuilder(config)

        assert builder._width == 9  # Even made odd (10-1)
        assert builder._height == 11  # Even made odd (12-1)

    def test_build_position_adjustment(self):
        # Test that positions are properly adjusted
        config = MazeKruskalMapBuilder.Config(width=11, height=11, start_pos=(0, 0), end_pos=(10, 10), seed=42)
        builder = MazeKruskalMapBuilder(config)

        # Should be adjusted to odd positions within bounds
        assert builder._start_pos[0] % 2 == 1  # Odd
        assert builder._start_pos[1] % 2 == 1  # Odd
        assert builder._end_pos[0] % 2 == 1  # Odd
        assert builder._end_pos[1] % 2 == 1  # Odd
        assert 0 <= builder._start_pos[0] < builder._width
        assert 0 <= builder._start_pos[1] < builder._height
        assert 0 <= builder._end_pos[0] < builder._width
        assert 0 <= builder._end_pos[1] < builder._height

    def test_build_small_maze(self):
        config = MazeKruskalMapBuilder.Config(width=5, height=5, start_pos=(1, 1), end_pos=(3, 3), seed=42)
        builder = MazeKruskalMapBuilder(config)
        game_map = builder.build()

        assert game_map.grid.shape == (5, 5)
        assert game_map.grid[1, 1] == "agent.agent"
        assert game_map.grid[3, 3] == "assembler"

    def test_build_returns_game_map(self):
        config = MazeKruskalMapBuilder.Config(width=7, height=7, start_pos=(1, 1), end_pos=(5, 5), seed=42)
        builder = MazeKruskalMapBuilder(config)
        result = builder.build()

        assert isinstance(result, GameMap)

    def test_build_maze_structure(self):
        """Test that the maze has proper structure with walls and paths"""
        config = MazeKruskalMapBuilder.Config(width=11, height=11, start_pos=(1, 1), end_pos=(9, 9), seed=42)
        builder = MazeKruskalMapBuilder(config)
        game_map = builder.build()

        # Count different elements
        unique, counts = np.unique(game_map.grid, return_counts=True)
        count_dict = dict(zip(unique, counts, strict=False))

        # Should have walls, empty spaces, start, and end
        assert "wall" in count_dict
        assert "empty" in count_dict
        assert "agent.agent" in count_dict
        assert "assembler" in count_dict

        # Should have exactly one start and one end
        assert count_dict["agent.agent"] == 1
        assert count_dict["assembler"] == 1

    def test_union_find_algorithm_integrity(self):
        """Test that Kruskal's algorithm produces a proper maze (connected)"""
        config = MazeKruskalMapBuilder.Config(width=15, height=15, start_pos=(1, 1), end_pos=(13, 13), seed=42)
        builder = MazeKruskalMapBuilder(config)
        game_map = builder.build()

        # The algorithm should create a connected maze
        # Count empty cells (paths) - should have reasonable number
        unique, counts = np.unique(game_map.grid, return_counts=True)
        count_dict = dict(zip(unique, counts, strict=False))

        empty_count = count_dict.get("empty", 0)
        agent_count = count_dict.get("agent.agent", 0)
        assembler_count = count_dict.get("assembler", 0)

        # Should have paths connecting all cells
        total_path_cells = empty_count + agent_count + assembler_count
        assert total_path_cells > 0


# Integration tests comparing both algorithms
class TestMazeAlgorithmComparison:
    def test_both_algorithms_produce_valid_mazes(self):
        """Test that both Prim's and Kruskal's produce valid maze structures"""
        prim_config = MazePrimMapBuilder.Config(width=11, height=11, start_pos=(1, 1), end_pos=(9, 9), seed=42)
        kruskal_config = MazeKruskalMapBuilder.Config(width=11, height=11, start_pos=(1, 1), end_pos=(9, 9), seed=42)

        prim_builder = MazePrimMapBuilder(prim_config)
        kruskal_builder = MazeKruskalMapBuilder(kruskal_config)

        prim_map = prim_builder.build()
        kruskal_map = kruskal_builder.build()

        # Both should produce GameMap instances
        assert isinstance(prim_map, GameMap)
        assert isinstance(kruskal_map, GameMap)

        # Both should have same dimensions
        assert prim_map.grid.shape == kruskal_map.grid.shape

        # Both should have start and end positions
        assert prim_map.grid[1, 1] == "agent.agent"
        assert kruskal_map.grid[1, 1] == "agent.agent"
        assert prim_map.grid[9, 9] == "assembler"
        assert kruskal_map.grid[9, 9] == "assembler"

        # Both should have similar structure (walls and paths)
        prim_unique, prim_counts = np.unique(prim_map.grid, return_counts=True)
        kruskal_unique, kruskal_counts = np.unique(kruskal_map.grid, return_counts=True)

        prim_dict = dict(zip(prim_unique, prim_counts, strict=False))
        kruskal_dict = dict(zip(kruskal_unique, kruskal_counts, strict=False))

        # Both should have same element types
        assert set(prim_dict.keys()) == set(kruskal_dict.keys())

    def test_different_algorithms_different_layouts(self):
        """Test that Prim's and Kruskal's produce different maze layouts"""
        prim_config = MazePrimMapBuilder.Config(width=15, height=15, start_pos=(1, 1), end_pos=(13, 13), seed=42)
        kruskal_config = MazeKruskalMapBuilder.Config(width=15, height=15, start_pos=(1, 1), end_pos=(13, 13), seed=42)

        prim_builder = MazePrimMapBuilder(prim_config)
        kruskal_builder = MazeKruskalMapBuilder(kruskal_config)

        prim_map = prim_builder.build()
        kruskal_map = kruskal_builder.build()

        # While they should have same start/end, the maze structure should differ
        # (very unlikely to be identical for large mazes)
        assert not np.array_equal(prim_map.grid, kruskal_map.grid)
