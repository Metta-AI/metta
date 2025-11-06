import numpy as np
import pytest

import mettagrid.map_builder.map_builder
import mettagrid.mapgen.types


class TestGameMap:
    def test_init(self):
        grid = np.array([["wall", "empty"], ["agent.agent", "altar"]], dtype=mettagrid.mapgen.types.map_grid_dtype)
        game_map = mettagrid.map_builder.map_builder.GameMap(grid)
        assert np.array_equal(game_map.grid, grid)
        assert game_map.grid.dtype == mettagrid.mapgen.types.map_grid_dtype

    def test_with_different_grid_sizes(self):
        # Test 1x1 grid
        small_grid = np.array([["wall"]], dtype=mettagrid.mapgen.types.map_grid_dtype)
        small_map = mettagrid.map_builder.map_builder.GameMap(small_grid)
        assert small_map.grid.shape == (1, 1)

        # Test larger grid
        large_grid = np.full((10, 15), "empty", dtype=mettagrid.mapgen.types.map_grid_dtype)
        large_map = mettagrid.map_builder.map_builder.GameMap(large_grid)
        assert large_map.grid.shape == (10, 15)


class ConcreteMapBuilderConfig(mettagrid.map_builder.map_builder.MapBuilderConfig["ConcreteMapBuilder"]):
    """Test implementation of abstract MapBuilderConfig"""

    pass


class ConcreteMapBuilder(mettagrid.map_builder.map_builder.MapBuilder[ConcreteMapBuilderConfig]):
    """Test implementation of abstract MapBuilder"""

    def build(self) -> mettagrid.map_builder.map_builder.GameMap:
        grid = np.array([["wall", "empty"], ["agent.agent", "altar"]], dtype=mettagrid.mapgen.types.map_grid_dtype)
        return mettagrid.map_builder.map_builder.GameMap(grid)


class TestMapBuilderConfig:
    def test_create_abstract_method(self):
        config = ConcreteMapBuilder.Config()
        builder = config.create()
        assert isinstance(builder, ConcreteMapBuilder)


class TestMapBuilder:
    def test_init(self):
        config = ConcreteMapBuilder.Config()
        builder = ConcreteMapBuilder(config)
        assert builder.config == config

    def test_build_abstract_method(self):
        config = ConcreteMapBuilder.Config()
        builder = ConcreteMapBuilder(config)
        game_map = builder.build()
        assert isinstance(game_map, mettagrid.map_builder.map_builder.GameMap)
        expected_grid = np.array(
            [["wall", "empty"], ["agent.agent", "altar"]], dtype=mettagrid.mapgen.types.map_grid_dtype
        )
        assert np.array_equal(game_map.grid, expected_grid)

    def test_map_builder_without_generic_parameter(self):
        with pytest.raises(
            TypeError,
            match=r"MapBuilderWithoutGenericParameter must inherit from MapBuilder",
        ):

            class MapBuilderWithoutGenericParameter(mettagrid.map_builder.map_builder.MapBuilder):
                pass


class TestMapGrid:
    def test_map_grid_dtype(self):
        # Test the dtype constant
        assert mettagrid.mapgen.types.map_grid_dtype == np.dtype("<U20")

        # Test creating array with correct dtype
        grid = np.array([["wall", "empty"]], dtype=mettagrid.mapgen.types.map_grid_dtype)
        assert grid.dtype == mettagrid.mapgen.types.map_grid_dtype

        # Test it can hold strings up to 20 characters
        long_name = "a" * 20
        grid = np.array([[long_name]], dtype=mettagrid.mapgen.types.map_grid_dtype)
        assert grid[0, 0] == long_name
