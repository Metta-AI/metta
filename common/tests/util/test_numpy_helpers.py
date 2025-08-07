"""Tests for metta.common.util.numpy_helpers module."""

import numpy as np

from metta.common.util.numpy_helpers import clean_numpy_types


class TestCleanNumpyTypes:
    """Test cases for the clean_numpy_types function."""

    def test_clean_numpy_scalar_array(self):
        """Test cleaning single-element numpy arrays."""
        # Integer array
        int_array = np.array([42])
        result = clean_numpy_types(int_array)
        assert result == 42
        assert isinstance(result, int)

        # Float array
        float_array = np.array([3.14])
        result = clean_numpy_types(float_array)
        assert result == 3.14
        assert isinstance(result, float)

    def test_clean_numpy_multi_element_array(self):
        """Test cleaning multi-element numpy arrays."""
        array = np.array([1, 2, 3, 4])
        result = clean_numpy_types(array)
        assert result == [1, 2, 3, 4]
        assert isinstance(result, list)

    def test_clean_numpy_generic_types(self):
        """Test cleaning numpy generic types."""
        # Integer type
        np_int = np.int64(42)
        result = clean_numpy_types(np_int)
        assert result == 42
        assert isinstance(result, int)

        # Float type
        np_float = np.float64(3.14)
        result = clean_numpy_types(np_float)
        assert result == 3.14
        assert isinstance(result, float)

    def test_clean_numpy_boolean_preservation(self):
        """Test that boolean types are preserved correctly."""
        # Boolean should not be converted to int
        bool_val = np.bool_(True)
        result = clean_numpy_types(bool_val)
        assert result is True
        assert isinstance(result, bool)

        bool_array = np.array([True])
        result = clean_numpy_types(bool_array)
        assert result is True
        assert isinstance(result, bool)

    def test_clean_dict_with_numpy_values(self):
        """Test cleaning dictionaries containing numpy values."""
        data = {
            "int_val": np.int32(10),
            "float_val": np.float32(2.5),
            "array_val": np.array([1, 2, 3]),
            "scalar_array": np.array([42]),
            "regular_val": "unchanged"
        }

        result = clean_numpy_types(data)

        assert result["int_val"] == 10
        assert isinstance(result["int_val"], int)
        assert result["float_val"] == 2.5
        assert isinstance(result["float_val"], float)
        assert result["array_val"] == [1, 2, 3]
        assert result["scalar_array"] == 42
        assert isinstance(result["scalar_array"], int)
        assert result["regular_val"] == "unchanged"

    def test_clean_list_with_numpy_values(self):
        """Test cleaning lists containing numpy values."""
        data = [
            np.int64(5),
            np.float64(1.5),
            np.array([10]),
            np.array([1, 2, 3]),
            "regular_string"
        ]

        result = clean_numpy_types(data)

        assert result[0] == 5
        assert isinstance(result[0], int)
        assert result[1] == 1.5
        assert isinstance(result[1], float)
        assert result[2] == 10
        assert isinstance(result[2], int)
        assert result[3] == [1, 2, 3]
        assert result[4] == "regular_string"

    def test_clean_nested_structures(self):
        """Test cleaning deeply nested structures."""
        data = {
            "level1": {
                "level2": [
                    {"numpy_int": np.int32(100)},
                    {"numpy_array": np.array([1.1, 2.2])},
                    "regular_item"
                ]
            },
            "numpy_list": [np.float64(x) for x in [1.0, 2.0, 3.0]]
        }

        result = clean_numpy_types(data)

        assert result["level1"]["level2"][0]["numpy_int"] == 100
        assert isinstance(result["level1"]["level2"][0]["numpy_int"], int)
        assert result["level1"]["level2"][1]["numpy_array"] == [1.1, 2.2]
        assert result["level1"]["level2"][2] == "regular_item"
        assert result["numpy_list"] == [1.0, 2.0, 3.0]
        assert all(isinstance(x, float) for x in result["numpy_list"])

    def test_clean_regular_python_types(self):
        """Test that regular Python types are unchanged."""
        data = {
            "int": 42,
            "float": 3.14,
            "string": "hello",
            "list": [1, 2, 3],
            "dict": {"nested": True},
            "none": None,
            "bool": True
        }

        result = clean_numpy_types(data)

        # Should be identical
        assert result == data
        assert result is not data  # But a new dict

    def test_clean_empty_structures(self):
        """Test cleaning empty structures."""
        assert clean_numpy_types({}) == {}
        assert clean_numpy_types([]) == []
        assert clean_numpy_types("") == ""
        assert clean_numpy_types(None) is None

    def test_clean_numpy_array_edge_cases(self):
        """Test edge cases with numpy arrays."""
        # Empty array
        empty_array = np.array([])
        result = clean_numpy_types(empty_array)
        assert result == []

        # 2D array
        array_2d = np.array([[1, 2], [3, 4]])
        result = clean_numpy_types(array_2d)
        assert result == [[1, 2], [3, 4]]

        # Array with mixed types (will be converted to common type by numpy)
        mixed_array = np.array([1, 2.5, 3])  # Becomes float array
        result = clean_numpy_types(mixed_array)
        assert result == [1.0, 2.5, 3.0]

    def test_clean_various_numpy_dtypes(self):
        """Test cleaning various numpy data types."""
        # Test different integer types
        for dtype in [np.int8, np.int16, np.int32, np.int64]:
            val = dtype(42)
            result = clean_numpy_types(val)
            assert result == 42
            assert isinstance(result, int)

        # Test different float types
        for dtype in [np.float16, np.float32, np.float64]:
            val = dtype(3.14)
            result = clean_numpy_types(val)
            # Use appropriate tolerance for float16 precision
            tolerance = 1e-3 if dtype == np.float16 else 1e-6
            assert abs(result - 3.14) < tolerance
            assert isinstance(result, float)

    def test_clean_numpy_strings(self):
        """Test cleaning numpy string types."""
        np_str = np.str_("hello")
        result = clean_numpy_types(np_str)
        assert result == "hello"
        assert isinstance(result, str)

        # String array
        str_array = np.array(["hello"])
        result = clean_numpy_types(str_array)
        assert result == "hello"
        assert isinstance(result, str)

    def test_clean_preserves_structure_independence(self):
        """Test that cleaning creates independent structures."""
        original = {
            "data": np.array([1, 2, 3]),
            "nested": {"value": np.int32(42)}
        }

        cleaned = clean_numpy_types(original)

        # Modify original
        original["data"][0] = 999
        original["nested"]["value"] = np.int32(999)

        # Cleaned should be unchanged
        assert cleaned["data"] == [1, 2, 3]
        assert cleaned["nested"]["value"] == 42

    def test_clean_performance_structure(self):
        """Test cleaning maintains reasonable performance for large structures."""
        # Create a moderately complex structure
        data = {
            f"key_{i}": {
                "array": np.array([j for j in range(10)]),
                "scalar": np.int64(i),
                "nested": [np.float32(i + 0.5) for _ in range(5)]
            }
            for i in range(10)
        }

        result = clean_numpy_types(data)

        # Verify structure is cleaned properly
        assert len(result) == 10
        for i in range(10):
            key = f"key_{i}"
            assert result[key]["array"] == list(range(10))
            assert result[key]["scalar"] == i
            assert isinstance(result[key]["scalar"], int)
            assert len(result[key]["nested"]) == 5
            assert all(isinstance(x, float) for x in result[key]["nested"])
