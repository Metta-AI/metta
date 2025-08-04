"""Tests for metta.common.util.datastruct module."""

from omegaconf import DictConfig, ListConfig

from metta.common.util.datastruct import convert_dict_to_cli_args, flatten_config


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


class TestConvertDictToCliArgs:
    """Test cases for the convert_dict_to_cli_args function."""

    def test_convert_simple_dict(self):
        """Test converting a simple dictionary to CLI arguments."""
        input_dict = {"a": 1, "b": "test", "c": True}
        expected = ["++a=1", "++b=test", "++c=true"]
        result = convert_dict_to_cli_args(input_dict)
        assert result == expected

    def test_convert_nested_dict(self):
        """Test converting a nested dictionary to CLI arguments."""
        input_dict = {"foo": {"bar": 1, "baz": "test"}}
        expected = ["++foo.bar=1", "++foo.baz=test"]
        result = convert_dict_to_cli_args(input_dict)
        assert result == expected

    def test_convert_with_prefix(self):
        """Test converting with a prefix."""
        input_dict = {"a": 1, "b": "test"}
        expected = ["++prefix.a=1", "++prefix.b=test"]
        result = convert_dict_to_cli_args(input_dict, prefix="prefix")
        assert result == expected

    def test_convert_boolean_values(self):
        """Test converting boolean values."""
        input_dict = {"enabled": True, "disabled": False}
        expected = ["++enabled=true", "++disabled=false"]
        result = convert_dict_to_cli_args(input_dict)
        assert result == expected

    def test_convert_numeric_values(self):
        """Test converting numeric values."""
        input_dict = {"int_val": 42, "float_val": 3.14, "small_float": 1e-8, "large_float": 1e8}
        result = convert_dict_to_cli_args(input_dict)
        assert "++int_val=42" in result
        assert "++float_val=3.14" in result
        assert "++small_float=1.000000e-08" in result  # Scientific notation for small numbers
        assert "++large_float=1.000000e+08" in result  # Scientific notation for large numbers

    def test_convert_none_value(self):
        """Test converting None values."""
        input_dict = {"null_value": None}
        expected = ["++null_value=null"]
        result = convert_dict_to_cli_args(input_dict)
        assert result == expected

    def test_convert_string_with_spaces(self):
        """Test converting strings with spaces."""
        input_dict = {"name": "John Doe", "description": "A test description"}
        expected = ["++name='John Doe'", "++description='A test description'"]
        result = convert_dict_to_cli_args(input_dict)
        assert result == expected

    def test_convert_string_with_equals(self):
        """Test converting strings containing equals signs."""
        input_dict = {"config": "key=value", "path": "file=name.txt"}
        expected = ["++config='key=value'", "++path='file=name.txt'"]
        result = convert_dict_to_cli_args(input_dict)
        assert result == expected

    def test_convert_deeply_nested_dict(self):
        """Test converting a deeply nested dictionary."""
        input_dict = {"level1": {"level2": {"level3": {"value": 42, "enabled": True}}}}
        expected = ["++level1.level2.level3.value=42", "++level1.level2.level3.enabled=true"]
        result = convert_dict_to_cli_args(input_dict)
        assert result == expected

    def test_convert_empty_dict(self):
        """Test converting an empty dictionary."""
        input_dict = {}
        expected = []
        result = convert_dict_to_cli_args(input_dict)
        assert result == expected

    def test_convert_mixed_types(self):
        """Test converting a dictionary with mixed types."""
        input_dict = {
            "string": "hello",
            "integer": 123,
            "float": 45.67,
            "boolean": False,
            "none": None,
            "nested": {"key": "value"},
        }
        result = convert_dict_to_cli_args(input_dict)
        assert "++string=hello" in result
        assert "++integer=123" in result
        assert "++float=45.67" in result
        assert "++boolean=false" in result
        assert "++none=null" in result
        assert "++nested.key=value" in result

    def test_convert_with_empty_prefix(self):
        """Test converting with an empty prefix (should behave same as no prefix)."""
        input_dict = {"a": 1, "b": "test"}
        expected = ["++a=1", "++b=test"]
        result = convert_dict_to_cli_args(input_dict, prefix="")
        assert result == expected

    def test_convert_complex_nested_structure(self):
        """Test converting a complex nested structure."""
        input_dict = {
            "trainer": {"learning_rate": 1e-4, "batch_size": 32, "optimizer": {"type": "adam", "weight_decay": 1e-5}},
            "sim": {"max_steps": 1000, "enabled": True},
        }
        result = convert_dict_to_cli_args(input_dict)
        assert "++trainer.learning_rate=0.0001" in result
        assert "++trainer.batch_size=32" in result
        assert "++trainer.optimizer.type=adam" in result
        assert "++trainer.optimizer.weight_decay=1e-05" in result
        assert "++sim.max_steps=1000" in result
        assert "++sim.enabled=true" in result
