"""
Comprehensive test suite for map format migration system.
"""

import numpy as np
import pytest

from metta.mettagrid.map_builder.map_builder import (
    GameMap,
    create_int_grid,
    create_legacy_grid,
)
from metta.mettagrid.migration.map_format_converter import MapFormatConverter
from metta.mettagrid.migration.performance import PerformanceBenchmark
from metta.mettagrid.migration.validation import MapFormatValidator
from metta.mettagrid.object_types import ObjectTypes


class TestMapFormatConverter:
    """Test suite for MapFormatConverter class."""

    @pytest.fixture
    def converter(self):
        """Create a MapFormatConverter instance for testing."""
        return MapFormatConverter()

    @pytest.fixture
    def sample_legacy_map(self):
        """Create a sample legacy map for testing."""
        legacy_grid = create_legacy_grid(5, 5)
        legacy_grid[:] = "empty"
        legacy_grid[0, :] = "wall"
        legacy_grid[-1, :] = "wall"
        legacy_grid[:, 0] = "wall"
        legacy_grid[:, -1] = "wall"
        legacy_grid[2, 2] = "agent"
        return legacy_grid

    @pytest.fixture
    def sample_int_map(self):
        """Create a sample int map for testing."""
        int_grid = create_int_grid(5, 5)
        int_grid[:] = ObjectTypes.EMPTY
        int_grid[0, :] = ObjectTypes.WALL
        int_grid[-1, :] = ObjectTypes.WALL
        int_grid[:, 0] = ObjectTypes.WALL
        int_grid[:, -1] = ObjectTypes.WALL
        int_grid[2, 2] = ObjectTypes.AGENT_BASE
        decoder_key = ["empty", "wall"] + [f"type_{i}" for i in range(2, ObjectTypes.AGENT_BASE)] + ["agent"]
        return int_grid, decoder_key

    def test_legacy_to_int_conversion(self, converter, sample_legacy_map):
        """Test converting legacy map to int format."""
        int_grid, decoder_key = converter.legacy_to_int(sample_legacy_map)

        assert int_grid.dtype == np.uint8
        assert int_grid.shape == sample_legacy_map.shape
        assert isinstance(decoder_key, list)
        assert "empty" in decoder_key
        assert "wall" in decoder_key
        assert "agent" in decoder_key

        # Verify specific conversions
        assert int_grid[1, 1] == ObjectTypes.EMPTY  # Interior should be empty
        assert int_grid[0, 0] == ObjectTypes.WALL  # Border should be wall
        assert int_grid[2, 2] == ObjectTypes.AGENT_BASE  # Agent position

    def test_int_to_legacy_conversion(self, converter, sample_int_map):
        """Test converting int map to legacy format."""
        int_grid, decoder_key = sample_int_map
        legacy_grid = converter.int_to_legacy(int_grid, decoder_key)

        assert legacy_grid.dtype.kind == "U"  # Unicode string
        assert legacy_grid.shape == int_grid.shape

        # Verify specific conversions
        assert legacy_grid[1, 1] == "empty"
        assert legacy_grid[0, 0] == "wall"
        assert legacy_grid[2, 2] == "agent"

    def test_round_trip_conversion(self, converter, sample_legacy_map):
        """Test round-trip conversion preserves data."""
        # Legacy -> Int -> Legacy
        int_grid, decoder_key = converter.legacy_to_int(sample_legacy_map)
        restored_legacy = converter.int_to_legacy(int_grid, decoder_key)

        assert np.array_equal(sample_legacy_map, restored_legacy)

    def test_game_map_conversion(self, converter, sample_legacy_map, sample_int_map):
        """Test GameMap conversion methods."""
        # Test legacy to int GameMap conversion
        legacy_game_map = GameMap(grid=sample_legacy_map)
        int_game_map = converter.convert_game_map_to_int(legacy_game_map)

        assert int_game_map.is_int_based()
        assert int_game_map.decoder_key is not None

        # Test int to legacy GameMap conversion
        int_grid, decoder_key = sample_int_map
        int_game_map = GameMap(grid=int_grid, decoder_key=decoder_key)
        legacy_game_map = converter.convert_game_map_to_legacy(int_game_map)

        assert legacy_game_map.is_legacy()

    def test_conversion_validation(self, converter, sample_legacy_map):
        """Test conversion validation functionality."""
        original = GameMap(grid=sample_legacy_map)
        converted = converter.convert_game_map_to_int(original)

        validation_results = converter.validate_conversion_integrity(original, converted)

        assert validation_results["shape_match"] is True
        assert validation_results["content_match"] is True
        assert len(validation_results["errors"]) == 0

    def test_batch_conversion(self, converter, sample_legacy_map, sample_int_map):
        """Test batch conversion of multiple maps."""
        int_grid, decoder_key = sample_int_map

        maps = [
            GameMap(grid=sample_legacy_map),
            GameMap(grid=int_grid, decoder_key=decoder_key),
        ]

        # Convert all to int format
        int_maps = converter.batch_convert_maps(maps, target_format="int")
        assert all(m.is_int_based() for m in int_maps)

        # Convert all to legacy format
        legacy_maps = converter.batch_convert_maps(maps, target_format="legacy")
        assert all(m.is_legacy() for m in legacy_maps)

    def test_error_handling(self, converter):
        """Test error handling in conversion."""
        # Test unknown object type
        bad_map = create_legacy_grid(3, 3)
        bad_map[:] = "unknown_object"

        with pytest.raises(ValueError):
            converter.legacy_to_int(bad_map)

        # Test invalid type_id
        int_grid = create_int_grid(3, 3)
        int_grid[:] = 255  # Invalid type_id
        decoder_key = ["empty", "wall"]

        with pytest.raises(ValueError):
            converter.int_to_legacy(int_grid, decoder_key)


class TestMapFormatValidator:
    """Test suite for MapFormatValidator class."""

    @pytest.fixture
    def validator(self):
        """Create a MapFormatValidator instance for testing."""
        return MapFormatValidator()

    def test_legacy_map_validation(self, validator):
        """Test validation of legacy maps."""
        # Valid map
        valid_map = create_legacy_grid(5, 5)
        valid_map[:] = "empty"
        valid_map[0, :] = "wall"

        results = validator.validate_legacy_map(valid_map)
        assert results["valid"] is True
        assert len(results["errors"]) == 0

        # Invalid map with unknown object
        invalid_map = create_legacy_grid(3, 3)
        invalid_map[:] = "unknown_type"

        results = validator.validate_legacy_map(invalid_map)
        assert results["valid"] is False
        assert len(results["errors"]) > 0

    def test_int_map_validation(self, validator):
        """Test validation of int-based maps."""
        # Valid map
        valid_map = create_int_grid(5, 5)
        valid_map[:] = ObjectTypes.EMPTY
        valid_map[0, :] = ObjectTypes.WALL
        decoder_key = ["empty", "wall", "agent"]

        results = validator.validate_int_map(valid_map, decoder_key)
        assert results["valid"] is True
        assert len(results["errors"]) == 0

        # Invalid map with out-of-range type_id
        invalid_map = create_int_grid(3, 3)
        invalid_map[:] = 255  # Out of decoder range

        results = validator.validate_int_map(invalid_map, decoder_key)
        assert results["valid"] is False
        assert len(results["errors"]) > 0

    def test_game_map_validation(self, validator):
        """Test validation of GameMap instances."""
        # Valid legacy GameMap
        legacy_grid = create_legacy_grid(3, 3)
        legacy_grid[:] = "empty"
        legacy_map = GameMap(grid=legacy_grid)

        results = validator.validate_game_map(legacy_map)
        assert results["valid"] is True
        assert results["format"] == "legacy"

        # Valid int GameMap
        int_grid = create_int_grid(3, 3)
        int_grid[:] = ObjectTypes.EMPTY
        decoder_key = ["empty", "wall"]
        int_map = GameMap(grid=int_grid, decoder_key=decoder_key)

        results = validator.validate_game_map(int_map)
        assert results["valid"] is True
        assert results["format"] == "int"

    def test_decoder_key_validation(self, validator):
        """Test decoder key validation."""
        # Valid decoder key
        valid_key = ["empty", "wall", "agent"]
        results = validator.validate_decoder_key(valid_key)
        assert results["valid"] is True

        # Invalid decoder key (missing empty)
        invalid_key = ["wall", "agent"]
        results = validator.validate_decoder_key(invalid_key)
        assert results["valid"] is False

        # Invalid decoder key (duplicates)
        duplicate_key = ["empty", "wall", "wall"]
        results = validator.validate_decoder_key(duplicate_key)
        assert results["valid"] is False

    def test_round_trip_validation(self, validator):
        """Test round-trip conversion consistency validation."""
        converter = MapFormatConverter()

        # Create original map
        original_grid = create_legacy_grid(3, 3)
        original_grid[:] = "empty"
        original_grid[1, 1] = "wall"
        original = GameMap(grid=original_grid)

        # Convert and convert back
        converted = converter.convert_game_map_to_int(original)
        converted_back = converter.convert_game_map_to_legacy(converted)

        results = validator.validate_conversion_consistency(original, converted, converted_back)
        assert results["valid"] is True
        assert results["round_trip_consistent"] is True


class TestPerformanceBenchmark:
    """Test suite for PerformanceBenchmark class."""

    @pytest.fixture
    def benchmark(self):
        """Create a PerformanceBenchmark instance for testing."""
        return PerformanceBenchmark()

    def test_map_creation_benchmark(self, benchmark):
        """Test map creation benchmarking."""
        results = benchmark.benchmark_map_creation(heights=[10, 20], widths=[10, 20], iterations=2)

        assert "creation" in results
        assert "10x10" in results["creation"]
        assert "20x20" in results["creation"]

    def test_conversion_benchmark(self, benchmark):
        """Test format conversion benchmarking."""
        # Create test maps
        legacy_grid = create_legacy_grid(10, 10)
        legacy_grid[:] = "empty"
        test_maps = [GameMap(grid=legacy_grid)]

        results = benchmark.benchmark_format_conversion(test_maps, iterations=2)
        assert "conversion" in results

    def test_memory_benchmark(self, benchmark):
        """Test memory usage benchmarking."""
        results = benchmark.benchmark_memory_usage([(10, 10), (20, 20)])

        assert "memory" in results
        assert "10x10" in results["memory"]

        # Check that int format uses less memory
        memory_10x10 = results["memory"]["10x10"]
        assert memory_10x10["savings"]["absolute"] > 0

    def test_comprehensive_benchmark(self, benchmark):
        """Test comprehensive benchmark suite."""
        # Create small test maps for faster testing
        legacy_grid = create_legacy_grid(5, 5)
        legacy_grid[:] = "empty"
        test_maps = [GameMap(grid=legacy_grid)]

        results = benchmark.run_comprehensive_benchmark(test_maps)

        assert "timestamp" in results
        assert "system_info" in results
        assert "creation" in results
        assert "memory" in results


class TestIntegration:
    """Integration tests for the migration system."""

    def test_complete_migration_workflow(self):
        """Test complete migration workflow from legacy to int format."""
        # Create a variety of legacy maps
        maps = []

        # Simple map
        simple_grid = create_legacy_grid(5, 5)
        simple_grid[:] = "empty"
        simple_grid[0, :] = "wall"
        maps.append(GameMap(grid=simple_grid))

        # Complex map with agents
        complex_grid = create_legacy_grid(10, 10)
        complex_grid[:] = "empty"
        complex_grid[0, :] = "wall"
        complex_grid[-1, :] = "wall"
        complex_grid[:, 0] = "wall"
        complex_grid[:, -1] = "wall"
        complex_grid[2, 2] = "agent"
        complex_grid[7, 7] = "agent"
        maps.append(GameMap(grid=complex_grid))

        # Initialize migration tools
        converter = MapFormatConverter()
        validator = MapFormatValidator()

        # Convert all maps to int format
        int_maps = converter.batch_convert_maps(maps, target_format="int")

        # Validate all conversions
        for original, converted in zip(maps, int_maps, strict=False):
            validation = converter.validate_conversion_integrity(original, converted)
            assert validation["valid"] is True
            assert validation["content_match"] is True

            # Validate individual maps
            orig_validation = validator.validate_game_map(original)
            conv_validation = validator.validate_game_map(converted)

            assert orig_validation["valid"] is True
            assert conv_validation["valid"] is True

        # Test round-trip consistency
        restored_maps = converter.batch_convert_maps(int_maps, target_format="legacy")

        for original, restored in zip(maps, restored_maps, strict=False):
            assert np.array_equal(original.grid, restored.grid)

    def test_performance_comparison(self):
        """Test performance comparison between formats."""
        benchmark = PerformanceBenchmark()

        # Create test maps of different sizes
        test_maps = []
        for size in [10, 50]:
            # Legacy map
            legacy_grid = create_legacy_grid(size, size)
            legacy_grid[:] = "empty"
            legacy_grid[0, :] = "wall"
            test_maps.append(GameMap(grid=legacy_grid))

            # Int map
            int_grid = create_int_grid(size, size)
            int_grid[:] = ObjectTypes.EMPTY
            int_grid[0, :] = ObjectTypes.WALL
            decoder_key = ["empty", "wall"]
            test_maps.append(GameMap(grid=int_grid, decoder_key=decoder_key))

        # Run performance benchmarks
        results = benchmark.benchmark_memory_usage([(10, 10), (50, 50)])

        # Verify memory savings
        for size_key in ["10x10", "50x50"]:
            memory_result = results["memory"][size_key]
            assert memory_result["savings"]["absolute"] > 0
            assert memory_result["savings"]["percentage"] > 0

    def test_error_recovery(self):
        """Test error recovery and graceful handling of edge cases."""
        converter = MapFormatConverter()
        validator = MapFormatValidator()

        # Test empty map
        empty_grid = create_legacy_grid(1, 1)
        empty_grid[:] = "empty"
        empty_map = GameMap(grid=empty_grid)

        validation = validator.validate_game_map(empty_map)
        assert validation["valid"] is True

        converted = converter.convert_game_map_to_int(empty_map)
        assert converted.is_int_based()

        # Test map with unknown objects (should fail gracefully)
        unknown_grid = create_legacy_grid(2, 2)
        unknown_grid[:] = "unknown_object"

        with pytest.raises(ValueError):
            converter.legacy_to_int(unknown_grid)


def run_migration_test_suite():
    """Run the complete migration test suite."""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_migration_test_suite()
