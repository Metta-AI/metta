"""Tests for clean_numpy_types function."""

import numpy as np

from metta.common.util.numpy_helpers import clean_numpy_types


class TestCleanNumpyTypes:
    """Test cases for clean_numpy_types function."""

    def test_numpy_scalar_conversion(self):
        """Test conversion of numpy scalar types to Python types."""
        # Test various numpy scalar types
        assert clean_numpy_types(np.int32(42)) == 42
        assert isinstance(clean_numpy_types(np.int32(42)), int)

        assert clean_numpy_types(np.float64(3.14)) == 3.14
        assert isinstance(clean_numpy_types(np.float64(3.14)), float)

        assert clean_numpy_types(np.bool_(True)) is True
        assert isinstance(clean_numpy_types(np.bool_(True)), bool)

    def test_numpy_array_conversion(self):
        """Test conversion of numpy arrays."""
        # Single element array should become scalar
        single_array = np.array([42])
        result = clean_numpy_types(single_array)
        assert result == 42
        assert isinstance(result, int)

        # Multi-element array should become list
        multi_array = np.array([1, 2, 3])
        result = clean_numpy_types(multi_array)
        assert result == [1, 2, 3]
        assert isinstance(result, list)
        assert all(isinstance(x, int) for x in result)

    def test_dict_conversion(self):
        """Test recursive conversion of dictionaries containing numpy types."""
        test_dict = {
            "int_val": np.int64(100),
            "float_val": np.float32(2.5),
            "array_val": np.array([1, 2, 3]),
            "nested_dict": {"inner_int": np.int16(50), "inner_array": np.array([4.0, 5.0])},
        }

        result = clean_numpy_types(test_dict)

        assert result["int_val"] == 100
        assert isinstance(result["int_val"], int)

        assert result["float_val"] == 2.5
        assert isinstance(result["float_val"], float)

        assert result["array_val"] == [1, 2, 3]
        assert isinstance(result["array_val"], list)

        assert result["nested_dict"]["inner_int"] == 50
        assert isinstance(result["nested_dict"]["inner_int"], int)

        assert result["nested_dict"]["inner_array"] == [4.0, 5.0]
        assert isinstance(result["nested_dict"]["inner_array"], list)

    def test_list_conversion(self):
        """Test recursive conversion of lists containing numpy types."""
        test_list = [np.int32(1), np.float64(2.5), [np.int16(3), np.float32(4.5)], {"key": np.int8(5)}]

        result = clean_numpy_types(test_list)

        assert result[0] == 1
        assert isinstance(result[0], int)

        assert result[1] == 2.5
        assert isinstance(result[1], float)

        assert result[2] == [3, 4.5]
        assert isinstance(result[2][0], int)
        assert isinstance(result[2][1], float)

        assert result[3]["key"] == 5
        assert isinstance(result[3]["key"], int)

    def test_non_numpy_types_unchanged(self):
        """Test that non-numpy types are returned unchanged."""
        # Python native types should pass through unchanged
        assert clean_numpy_types(42) == 42
        assert clean_numpy_types(3.14) == 3.14
        assert clean_numpy_types("hello") == "hello"
        assert clean_numpy_types(True) is True
        assert clean_numpy_types(None) is None

        # Complex structures with native types
        test_data = {
            "str": "test",
            "int": 123,
            "float": 45.6,
            "bool": False,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
        }

        result = clean_numpy_types(test_data)
        assert result == test_data

    def test_mixed_data_structure(self):
        """Test conversion of complex mixed data structures."""
        test_data = {
            "parameters": {
                "learning_rate": np.float64(0.001),
                "batch_size": np.int32(64),
                "epochs": 100,  # Regular int
                "metrics": [
                    np.float32(0.95),
                    np.float32(0.87),
                    0.92,  # Regular float
                ],
            },
            "metadata": {"experiment_name": "test_run", "version": np.int16(1), "success": np.bool_(True)},
        }

        result = clean_numpy_types(test_data)

        # Check that numpy types were converted
        assert isinstance(result["parameters"]["learning_rate"], float)
        assert isinstance(result["parameters"]["batch_size"], int)
        assert isinstance(result["metadata"]["version"], int)
        assert isinstance(result["metadata"]["success"], bool)

        # Check that regular types were preserved
        assert isinstance(result["parameters"]["epochs"], int)
        assert isinstance(result["parameters"]["metrics"][2], float)
        assert isinstance(result["metadata"]["experiment_name"], str)

        # Check values are correct
        assert result["parameters"]["learning_rate"] == 0.001
        assert result["parameters"]["batch_size"] == 64
        assert result["metadata"]["version"] == 1
        assert result["metadata"]["success"] is True

    def test_empty_containers(self):
        """Test handling of empty containers."""
        assert clean_numpy_types({}) == {}
        assert clean_numpy_types([]) == []

    def test_numpy_array_2d(self):
        """Test conversion of 2D numpy arrays."""
        array_2d = np.array([[1, 2], [3, 4]])
        result = clean_numpy_types(array_2d)

        assert result == [[1, 2], [3, 4]]
        assert isinstance(result, list)
        assert all(isinstance(row, list) for row in result)
        assert all(isinstance(val, int) for row in result for val in row)
