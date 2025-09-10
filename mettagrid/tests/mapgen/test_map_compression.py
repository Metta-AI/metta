"""Test map compression utilities."""

import numpy as np
import pytest

from metta.mettagrid.mapgen.utils.map_compression import MapCompressor, compress_map, decompress_map


class TestMapCompressor:
    """Test map compression functionality."""

    def test_basic_compression(self):
        """Test basic string grid to byte grid compression."""
        # Create a simple test grid
        string_grid = np.array(
            [["wall", "wall", "empty"], ["wall", "agent.team_1", "empty"], ["wall", "generator", "empty"]], dtype="<U20"
        )

        # Compress
        compressor = MapCompressor()
        byte_grid, object_key = compressor.compress(string_grid)

        # Verify shape is preserved
        assert byte_grid.shape == string_grid.shape

        # Verify byte grid is uint8
        assert byte_grid.dtype == np.uint8

        # Verify object key contains all unique objects (sorted)
        assert object_key == ["agent.team_1", "empty", "generator", "wall"]

        # Verify byte values are correct indices
        assert byte_grid[0, 0] == 3  # wall
        assert byte_grid[0, 2] == 1  # empty
        assert byte_grid[1, 1] == 0  # agent.team_1
        assert byte_grid[2, 1] == 2  # generator

    def test_decompression(self):
        """Test round-trip compression and decompression."""
        # Create a test grid
        string_grid = np.array(
            [["wall", "agent.team_1", "generator"], ["empty", "wall", "agent.team_2"], ["converter", "altar", "wall"]],
            dtype="<U20",
        )

        # Compress and decompress
        compressor = MapCompressor()
        byte_grid, object_key = compressor.compress(string_grid)
        decompressed_grid = compressor.decompress(byte_grid, object_key)

        # Verify exact reconstruction
        np.testing.assert_array_equal(string_grid, decompressed_grid)

    def test_validation_with_valid_objects(self):
        """Test validation catches unknown objects."""
        # Create a grid with known objects
        string_grid = np.array([["wall", "empty"], ["agent.team_1", "generator"]], dtype="<U20")

        # Set up validator with valid objects
        valid_objects = {"wall", "agent.team_1", "generator"}  # Note: "empty" not included
        compressor = MapCompressor(valid_objects)

        # This should succeed - "empty" is always allowed
        byte_grid, object_key = compressor.compress(string_grid)
        assert len(object_key) == 4

    def test_validation_with_unknown_objects(self):
        """Test validation fails on unknown objects."""
        # Create a grid with unknown object
        string_grid = np.array([["wall", "unknown_object"], ["agent.team_1", "generator"]], dtype="<U20")

        # Set up validator
        valid_objects = {"wall", "agent.team_1", "generator"}
        compressor = MapCompressor(valid_objects)

        # This should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            compressor.compress(string_grid)

        assert "unknown_object" in str(exc_info.value)
        assert "Valid objects:" in str(exc_info.value)

    def test_convenience_functions(self):
        """Test convenience function wrappers."""
        string_grid = np.array([["wall", "empty"]], dtype="<U20")

        # Test compress_map
        byte_grid, object_key = compress_map(string_grid)
        assert byte_grid.shape == string_grid.shape

        # Test decompress_map
        decompressed = decompress_map(byte_grid, object_key)
        np.testing.assert_array_equal(string_grid, decompressed)

    def test_large_grid_performance(self):
        """Test compression works efficiently on larger grids."""
        # Create a 100x100 grid with various objects
        size = 100
        objects = ["empty", "wall", "agent.team_1", "agent.team_2", "generator", "converter", "altar"]

        # Generate random grid
        rng = np.random.default_rng(42)
        indices = rng.integers(0, len(objects), size=(size, size))
        string_grid = np.array([[objects[idx] for idx in row] for row in indices], dtype="<U20")

        # Compress
        compressor = MapCompressor()
        byte_grid, object_key = compressor.compress(string_grid)

        # Verify compression ratio
        string_size = string_grid.nbytes
        byte_size = byte_grid.nbytes + sum(len(s) for s in object_key)
        compression_ratio = string_size / byte_size

        # Should achieve at least 5x compression
        assert compression_ratio > 5.0

        # Verify round-trip
        decompressed = compressor.decompress(byte_grid, object_key)
        np.testing.assert_array_equal(string_grid, decompressed)

    def test_empty_grid(self):
        """Test handling of empty grid."""
        string_grid = np.array([], dtype="<U20").reshape(0, 0)

        compressor = MapCompressor()
        byte_grid, object_key = compressor.compress(string_grid)

        assert byte_grid.shape == (0, 0)
        assert object_key == []

        # Round-trip should work
        decompressed = compressor.decompress(byte_grid, object_key)
        assert decompressed.shape == (0, 0)
