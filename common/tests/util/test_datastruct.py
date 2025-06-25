"""Tests for metta.common.util.datastruct module."""

from omegaconf import DictConfig, ListConfig

from metta.common.util.datastruct import flatten_config


class TestFlattenConfig:
    """Test cases for the flatten_config function."""

    def test_flatten_simple_dict(self):
        """Test flattening a simple dictionary."""
        input_dict = {"a": 1, "b": 2, "c": 3}
        expected = {"a": 1, "b": 2, "c": 3}
        result = flatten_config(input_dict)
        assert result == expected

    def test_flatten_nested_dict(self):
        """Test flattening a nested dictionary."""
        input_dict = {"foo": {"bar": 1, "baz": 2}, "qux": 3}
        expected = {"foo.bar": 1, "foo.baz": 2, "qux": 3}
        result = flatten_config(input_dict)
        assert result == expected

    def test_flatten_simple_list(self):
        """Test flattening a simple list."""
        input_list = [1, 2, 3]
        expected = {"0": 1, "1": 2, "2": 3}
        result = flatten_config(input_list)
        assert result == expected

    def test_flatten_list_with_dicts(self):
        """Test flattening a list containing dictionaries."""
        input_list = [{"a": 1}, {"b": 2}]
        expected = {"0.a": 1, "1.b": 2}
        result = flatten_config(input_list)
        assert result == expected

    def test_flatten_dict_with_list(self):
        """Test flattening a dictionary containing a list."""
        input_dict = {"foo": {"bar": [{"a": 1}, {"b": 2}]}}
        expected = {"foo.bar.0.a": 1, "foo.bar.1.b": 2}
        result = flatten_config(input_dict)
        assert result == expected

    def test_flatten_complex_nested_structure(self):
        """Test flattening a complex nested structure."""
        input_data = {
            "level1": {
                "level2": {
                    "items": [{"name": "item1", "value": 10}, {"name": "item2", "value": 20}],
                    "config": {"enabled": True, "settings": [1, 2, 3]},
                }
            },
            "simple": "value",
        }
        expected = {
            "level1.level2.items.0.name": "item1",
            "level1.level2.items.0.value": 10,
            "level1.level2.items.1.name": "item2",
            "level1.level2.items.1.value": 20,
            "level1.level2.config.enabled": True,
            "level1.level2.config.settings.0": 1,
            "level1.level2.config.settings.1": 2,
            "level1.level2.config.settings.2": 3,
            "simple": "value",
        }
        result = flatten_config(input_data)
        assert result == expected

    def test_flatten_empty_dict(self):
        """Test flattening an empty dictionary."""
        input_dict = {}
        expected = {}
        result = flatten_config(input_dict)
        assert result == expected

    def test_flatten_empty_list(self):
        """Test flattening an empty list."""
        input_list = []
        expected = {}
        result = flatten_config(input_list)
        assert result == expected

    def test_flatten_scalar_value(self):
        """Test flattening a scalar value."""
        result = flatten_config(42, parent_key="test")
        expected = {"test": 42}
        assert result == expected

    def test_flatten_scalar_value_no_parent_key(self):
        """Test flattening a scalar value without parent key."""
        result = flatten_config(42)
        expected = {"": 42}
        assert result == expected

    def test_flatten_with_custom_separator(self):
        """Test flattening with a custom separator."""
        input_dict = {"foo": {"bar": [1, 2]}}
        expected = {"foo_bar_0": 1, "foo_bar_1": 2}
        result = flatten_config(input_dict, sep="_")
        assert result == expected

    def test_flatten_with_parent_key(self):
        """Test flattening with a parent key."""
        input_dict = {"a": 1, "b": 2}
        expected = {"prefix.a": 1, "prefix.b": 2}
        result = flatten_config(input_dict, parent_key="prefix")
        assert result == expected

    def test_flatten_omegaconf_dictconfig(self):
        """Test flattening an OmegaConf DictConfig."""
        input_config = DictConfig({"foo": {"bar": 1, "baz": 2}})
        expected = {"foo.bar": 1, "foo.baz": 2}
        result = flatten_config(input_config)
        assert result == expected

    def test_flatten_omegaconf_listconfig(self):
        """Test flattening an OmegaConf ListConfig."""
        input_config = ListConfig([{"a": 1}, {"b": 2}])
        expected = {"0.a": 1, "1.b": 2}
        result = flatten_config(input_config)
        assert result == expected

    def test_flatten_mixed_omegaconf_and_native(self):
        """Test flattening a mix of OmegaConf and native Python structures."""
        input_data = {
            "native_dict": {"a": 1},
            "omega_dict": DictConfig({"b": 2}),
            "native_list": [3, 4],
            "omega_list": ListConfig([5, 6]),
        }
        expected = {
            "native_dict.a": 1,
            "omega_dict.b": 2,
            "native_list.0": 3,
            "native_list.1": 4,
            "omega_list.0": 5,
            "omega_list.1": 6,
        }
        result = flatten_config(input_data)
        assert result == expected

    def test_flatten_nested_empty_structures(self):
        """Test flattening structures containing empty nested structures."""
        input_data = {"empty_dict": {}, "empty_list": [], "nested": {"also_empty": {}, "list_empty": []}}
        expected = {}
        result = flatten_config(input_data)
        assert result == expected

    def test_flatten_various_data_types(self):
        """Test flattening with various data types as values."""
        input_data = {
            "string": "hello",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "none": None,
            "nested": {"list_of_mixed": ["string", 123, False, None]},
        }
        expected = {
            "string": "hello",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "none": None,
            "nested.list_of_mixed.0": "string",
            "nested.list_of_mixed.1": 123,
            "nested.list_of_mixed.2": False,
            "nested.list_of_mixed.3": None,
        }
        result = flatten_config(input_data)
        assert result == expected

    def test_flatten_deeply_nested_structure(self):
        """Test flattening a deeply nested structure."""
        input_data = {"level1": {"level2": {"level3": {"level4": {"level5": [{"deep_value": "found"}]}}}}}
        expected = {"level1.level2.level3.level4.level5.0.deep_value": "found"}
        result = flatten_config(input_data)
        assert result == expected

    def test_flatten_list_of_lists(self):
        """Test flattening a list containing other lists."""
        input_data = [[1, 2], [3, 4], {"nested": [5, 6]}]
        expected = {"0.0": 1, "0.1": 2, "1.0": 3, "1.1": 4, "2.nested.0": 5, "2.nested.1": 6}
        result = flatten_config(input_data)
        assert result == expected
