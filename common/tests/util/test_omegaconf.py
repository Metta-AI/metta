"""Tests for metta.common.util.omegaconf module."""

from typing import Any, Dict

import pytest
from omegaconf import DictConfig, ListConfig, OmegaConf
from enum import Enum

from metta.common.util.omegaconf import convert_to_dict


class Color(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class TestConvertToDict:
    """Test cases for the convert_to_dict function."""

    def test_convert_empty_dictconfig(self):
        """Test converting an empty DictConfig."""
        config = OmegaConf.create({})
        result = convert_to_dict(config)

        assert result == {}
        assert isinstance(result, dict)
        assert not isinstance(result, DictConfig)

    def test_convert_simple_dictconfig(self):
        """Test converting a simple DictConfig with various types."""
        config_data = {
            "string_val": "hello",
            "int_val": 42,
            "float_val": 3.14,
            "bool_val": True,
            "none_val": None,
        }
        config = OmegaConf.create(config_data)
        result = convert_to_dict(config)

        assert result == config_data
        assert isinstance(result, dict)
        assert not isinstance(result, DictConfig)

        # Check types are preserved
        assert isinstance(result["string_val"], str)
        assert isinstance(result["int_val"], int)
        assert isinstance(result["float_val"], float)
        assert isinstance(result["bool_val"], bool)
        assert result["none_val"] is None

    def test_convert_nested_dictconfig(self):
        """Test converting a nested DictConfig structure."""
        config_data = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "credentials": {
                    "username": "admin",
                    "password": "secret",
                },
            },
            "features": {
                "enable_cache": True,
                "max_connections": 100,
            },
        }
        config = OmegaConf.create(config_data)
        result = convert_to_dict(config)

        assert result == config_data
        assert isinstance(result, dict)
        assert isinstance(result["database"], dict)
        assert isinstance(result["database"]["credentials"], dict)

    def test_convert_with_list_values(self):
        """Test converting DictConfig containing lists."""
        config_data = {
            "simple_list": [1, 2, 3],
            "string_list": ["a", "b", "c"],
            "nested_list": [
                {"name": "item1", "value": 10},
                {"name": "item2", "value": 20},
            ],
            "mixed_list": [1, "two", 3.0, True, None],
        }
        config = OmegaConf.create(config_data)
        result = convert_to_dict(config)

        assert result == config_data
        assert isinstance(result["simple_list"], list)
        assert isinstance(result["nested_list"][0], dict)

    def test_convert_with_interpolations_resolved(self):
        """Test converting DictConfig with interpolations (default resolve=True)."""
        config_data = {
            "base_url": "https://api.example.com",
            "version": "v1",
            "full_url": "${base_url}/${version}",
            "timeout": 30,
            "retry_timeout": "${timeout}",
        }
        config = OmegaConf.create(config_data)
        result = convert_to_dict(config)

        # Interpolations should be resolved
        expected = {
            "base_url": "https://api.example.com",
            "version": "v1",
            "full_url": "https://api.example.com/v1",
            "timeout": 30,
            "retry_timeout": 30,
        }
        assert result == expected

    def test_convert_with_interpolations_unresolved(self):
        """Test converting DictConfig with interpolations (resolve=False)."""
        config_data = {
            "base_url": "https://api.example.com",
            "version": "v1",
            "full_url": "${base_url}/${version}",
        }
        config = OmegaConf.create(config_data)
        result = convert_to_dict(config, resolve=False)

        # Interpolations should NOT be resolved
        assert result["base_url"] == "https://api.example.com"
        assert result["version"] == "v1"
        assert result["full_url"] == "${base_url}/${version}"

    def test_convert_with_enum_values(self):
        """Test converting DictConfig containing enum values."""
        config_data = {
            "primary_color": Color.RED,
            "secondary_color": Color.BLUE,
            "theme": {
                "accent": Color.GREEN,
            },
        }
        config = OmegaConf.create(config_data)
        result = convert_to_dict(config)

        # Enums should be converted to their name (not value)
        assert result["primary_color"] == "RED"
        assert result["secondary_color"] == "BLUE"
        assert result["theme"]["accent"] == "GREEN"

    def test_convert_with_integer_keys(self):
        """Test converting DictConfig with integer keys (converted to strings)."""
        config_data = {
            1: "first",
            2: "second",
            "string_key": "value",
            42: {"nested": "dict"},
        }
        config = OmegaConf.create(config_data)
        result = convert_to_dict(config)

        # All keys should be converted to strings
        expected = {
            "1": "first",
            "2": "second",
            "string_key": "value",
            "42": {"nested": "dict"},
        }
        assert result == expected

        # Verify all keys are strings
        for key in result.keys():
            assert isinstance(key, str)

    def test_convert_with_mixed_key_types(self):
        """Test converting DictConfig with various key types."""
        # OmegaConf can handle string keys primarily, but let's test edge cases
        config_data = {
            "str_key": "string value",
            "123": "numeric string key",
            "true": "boolean-like string key",
        }
        config = OmegaConf.create(config_data)
        result = convert_to_dict(config)

        assert result == config_data
        # All keys should already be strings or converted to strings
        for key in result.keys():
            assert isinstance(key, str)

    def test_convert_listconfig_raises_error(self):
        """Test that passing a ListConfig raises ValueError."""
        list_config = OmegaConf.create([1, 2, 3])

        with pytest.raises(ValueError) as exc_info:
            convert_to_dict(list_config)

        error_msg = str(exc_info.value)
        assert "Expected dictionary configuration" in error_msg
        assert "list" in error_msg

    def test_convert_none_config_raises_error(self):
        """Test that passing None raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            convert_to_dict(None)

        error_msg = str(exc_info.value)
        assert "Input cfg is not an OmegaConf config object" in error_msg
        assert "NoneType" in error_msg



    def test_convert_regular_dict_input_raises_error(self):
        """Test that regular Python dict raises error since it's not an OmegaConf object."""
        regular_dict = {
            "key1": "value1",
            "key2": {"nested": "value"},
            "key3": [1, 2, 3],
        }

        # Regular dicts are not OmegaConf objects, so should raise error
        with pytest.raises(ValueError) as exc_info:
            convert_to_dict(regular_dict)

        error_msg = str(exc_info.value)
        assert "Input cfg is not an OmegaConf config object" in error_msg
        assert "dict" in error_msg

    def test_return_type_annotation_correctness(self):
        """Test that the function returns the correct type as annotated."""
        config = OmegaConf.create({"test": "value"})
        result = convert_to_dict(config)

        # Check that the return type matches the annotation: Dict[str, Any]
        assert isinstance(result, dict)

        # All keys should be strings
        for key in result.keys():
            assert isinstance(key, str)

        # Values can be Any type
        config_with_various_types = OmegaConf.create({
            "str": "text",
            "int": 42,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "none": None,
        })
        result = convert_to_dict(config_with_various_types)

        # Verify the types are preserved correctly
        assert isinstance(result["str"], str)
        assert isinstance(result["int"], int)
        assert isinstance(result["list"], list)
        assert isinstance(result["dict"], dict)
        assert result["none"] is None



    def test_complex_real_world_scenario(self):
        """Test with a complex configuration similar to real-world usage."""
        config_data = {
            "model": {
                "name": "transformer",
                "layers": 12,
                "hidden_size": 768,
                "attention_heads": 12,
            },
            "training": {
                "batch_size": 32,
                "learning_rate": 0.001,
                "epochs": 100,
                "optimizer": "adam",
                "scheduler": {
                    "type": "cosine",
                    "warmup_steps": 1000,
                },
            },
            "data": {
                "train_path": "/data/train",
                "val_path": "/data/val",
                "preprocessing": ["normalize", "tokenize"],
            },
            "evaluation": {
                "metrics": ["accuracy", "f1", "bleu"],
                "save_predictions": True,
            },
        }

        config = OmegaConf.create(config_data)
        result = convert_to_dict(config)

        assert result == config_data
        assert isinstance(result, dict)

        # Test deep nesting is preserved
        assert result["training"]["scheduler"]["type"] == "cosine"
        assert result["data"]["preprocessing"] == ["normalize", "tokenize"]
        assert result["evaluation"]["save_predictions"] is True
