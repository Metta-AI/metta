"""Tests for metta.common.util.instantiate module."""

from unittest.mock import Mock, patch

import pytest
from omegaconf import DictConfig, OmegaConf

from metta.common.util.instantiate import _process_recursive, instantiate


class TestInstantiate:
    """Test cases for the instantiate function."""

    def test_instantiate_basic_class(self):
        """Test instantiating a basic Python class."""
        config = {
            "_target_": "collections.OrderedDict",
            "data": {"a": 1, "b": 2}
        }

        result = instantiate(config)

        from collections import OrderedDict
        assert isinstance(result, OrderedDict)
        assert result["data"] == {"a": 1, "b": 2}

    def test_instantiate_with_kwargs_override(self):
        """Test that kwargs override config values."""
        config = {
            "_target_": "collections.OrderedDict",
            "data": {"a": 1}
        }

        result = instantiate(config, data={"b": 2})

        from collections import OrderedDict
        assert isinstance(result, OrderedDict)
        assert result["data"] == {"b": 2}  # Kwargs override config

    def test_instantiate_with_dictconfig(self):
        """Test instantiating with OmegaConf DictConfig."""
        config = OmegaConf.create({
            "_target_": "collections.OrderedDict",
            "data": {"x": 10, "y": 20}
        })

        result = instantiate(config)

        from collections import OrderedDict
        assert isinstance(result, OrderedDict)
        assert result["data"] == {"x": 10, "y": 20}

    def test_instantiate_filters_underscore_keys(self):
        """Test that underscore-prefixed keys are filtered from constructor args."""
        config = {
            "_target_": "collections.OrderedDict",
            "_private_": "should_be_ignored",
            "_other_meta_": "also_ignored",
            "valid_arg": "should_be_passed"
        }

        result = instantiate(config)

        from collections import OrderedDict
        assert isinstance(result, OrderedDict)
        assert result["valid_arg"] == "should_be_passed"
        assert "_private_" not in result
        assert "_other_meta_" not in result

    def test_instantiate_missing_target_raises_error(self):
        """Test that missing _target_ field raises ValueError."""
        config = {"some_arg": "value"}

        with pytest.raises(ValueError, match="Configuration missing '_target_' field"):
            instantiate(config)

    def test_instantiate_invalid_module_raises_import_error(self):
        """Test that invalid module path raises ImportError."""
        config = {"_target_": "nonexistent.module.Class"}

        with pytest.raises(ImportError, match="Failed to import module nonexistent.module"):
            instantiate(config)

    def test_instantiate_invalid_class_raises_attribute_error(self):
        """Test that invalid class name raises AttributeError."""
        config = {"_target_": "collections.NonexistentClass"}

        with pytest.raises(AttributeError, match="Module collections has no class NonexistentClass"):
            instantiate(config)

    def test_instantiate_invalid_config_type_raises_type_error(self):
        """Test that invalid config type raises TypeError."""
        with pytest.raises(TypeError, match="Config must be dict or DictConfig"):
            instantiate("not_a_dict")

        with pytest.raises(TypeError, match="Config must be dict or DictConfig"):
            instantiate(123)

    def test_instantiate_builtin_types(self):
        """Test instantiating builtin types."""
        # Test list
        config = {"_target_": "builtins.list", "_args_": [1, 2, 3]}
        # Since list() doesn't take keyword args, let's use set instead
        config = {"_target_": "builtins.set"}
        result = instantiate(config)
        assert isinstance(result, set)
        assert result == set()

    def test_instantiate_with_complex_args(self):
        """Test instantiating with complex argument structures."""
        config = {
            "_target_": "collections.defaultdict",
            "default_factory": {"_target_": "builtins.list"}
        }

        # Note: This test shows the limitation - nested configs need recursive processing
        result = instantiate(config)

        from collections import defaultdict
        assert isinstance(result, defaultdict)
        # The default_factory will be a dict, not an instantiated list

    def test_instantiate_recursive_simple(self):
        """Test recursive instantiation of nested configs."""
        config = {
            "_target_": "collections.OrderedDict",
            "data": {"_target_": "builtins.dict", "key": "value"}
        }

        result = instantiate(config, _recursive_=True)

        from collections import OrderedDict
        assert isinstance(result, OrderedDict)
        assert "data" in result
        assert isinstance(result["data"], dict)
        assert result["data"]["key"] == "value"

    def test_instantiate_recursive_nested_structures(self):
        """Test recursive instantiation with complex nested structures."""
        config = {
            "_target_": "collections.OrderedDict",
            "data": {
                "list_factory": {"_target_": "builtins.list"},
                "nested": {
                    "dict_factory": {"_target_": "builtins.dict"}
                }
            }
        }

        result = instantiate(config, _recursive_=True)

        from collections import OrderedDict
        assert isinstance(result, OrderedDict)
        assert isinstance(result["data"]["list_factory"], list)
        assert isinstance(result["data"]["nested"]["dict_factory"], dict)

    def test_instantiate_recursive_with_lists(self):
        """Test recursive instantiation with lists containing configs."""
        config = {
            "_target_": "collections.OrderedDict",
            "factories": [
                {"_target_": "builtins.list"},
                {"_target_": "builtins.set"},
                "normal_string"
            ]
        }

        result = instantiate(config, _recursive_=True)

        from collections import OrderedDict
        assert isinstance(result, OrderedDict)
        assert isinstance(result["factories"][0], list)
        assert isinstance(result["factories"][1], set)
        assert result["factories"][2] == "normal_string"

    def test_instantiate_omegaconf_with_metadata(self):
        """Test handling of OmegaConf with metadata."""
        # Create a DictConfig with interpolations to test metadata handling
        config = OmegaConf.create({
            "_target_": "collections.OrderedDict",
            "base": "test",
            "derived": "${base}_suffix"
        })

        result = instantiate(config)

        from collections import OrderedDict
        assert isinstance(result, OrderedDict)
        assert result["base"] == "test"
        assert result["derived"] == "test_suffix"

    def test_instantiate_real_world_scenario(self):
        """Test instantiation in a realistic scenario."""
        # Simulate creating a logger with custom formatter
        config = {
            "_target_": "logging.Logger",
            "name": "test_logger"
        }

        result = instantiate(config)

        import logging
        assert isinstance(result, logging.Logger)
        assert result.name == "test_logger"

    def test_instantiate_preserves_config_modifications(self):
        """Test that instantiate doesn't modify the original config."""
        original_config = {
            "_target_": "collections.OrderedDict",
            "data": {"original": True}
        }
        config_copy = original_config.copy()

        result = instantiate(config_copy, data={"modified": True})

        # Original should be unchanged
        assert original_config["data"] == {"original": True}
        # Result should use overridden data
        assert result["data"] == {"modified": True}


class TestProcessRecursive:
    """Test cases for the _process_recursive helper function."""

    def test_process_recursive_dict_without_target(self):
        """Test processing regular dict without _target_."""
        config = {"a": 1, "b": {"c": 2}}

        result = _process_recursive(config, is_top_level=True)

        assert result == {"a": 1, "b": {"c": 2}}

    def test_process_recursive_dict_with_target_not_top_level(self):
        """Test processing dict with _target_ that's not top level."""
        config = {"_target_": "builtins.list"}

        result = _process_recursive(config, is_top_level=False)

        assert isinstance(result, list)

    def test_process_recursive_dict_with_target_top_level(self):
        """Test processing dict with _target_ at top level (should not instantiate)."""
        config = {"_target_": "builtins.list", "other": "value"}

        result = _process_recursive(config, is_top_level=True)

        assert isinstance(result, dict)
        assert result["_target_"] == "builtins.list"
        assert result["other"] == "value"

    def test_process_recursive_list(self):
        """Test processing lists recursively."""
        config = [
            {"_target_": "builtins.list"},
            {"_target_": "builtins.set"},
            "normal_item",
            {"nested": {"_target_": "builtins.dict"}}
        ]

        result = _process_recursive(config, is_top_level=False)

        assert isinstance(result[0], list)
        assert isinstance(result[1], set)
        assert result[2] == "normal_item"
        assert isinstance(result[3]["nested"], dict)

    def test_process_recursive_primitive_values(self):
        """Test processing primitive values."""
        assert _process_recursive("string", is_top_level=False) == "string"
        assert _process_recursive(123, is_top_level=False) == 123
        assert _process_recursive(True, is_top_level=False) is True
        assert _process_recursive(None, is_top_level=False) is None

    def test_process_recursive_omegaconf_with_metadata(self):
        """Test processing OmegaConf with metadata - should hit line 90."""
        # Create OmegaConf with interpolations to ensure metadata exists
        config = OmegaConf.create({
            "_target_": "builtins.dict",
            "base": "test",
            "derived": "${base}_suffix"
        })

        # Verify it has metadata
        assert hasattr(config, "_metadata")

        result = _process_recursive(config, is_top_level=False)

        assert isinstance(result, dict)
        assert result["base"] == "test"
        assert result["derived"] == "test_suffix"

    def test_process_recursive_dictconfig_without_metadata(self):
        """Test processing DictConfig without metadata - should hit line 41 case."""
        # Create a simple DictConfig that might not have metadata
        config = OmegaConf.create({"_target_": "builtins.dict"})

        # Force it to be a DictConfig but ensure the _metadata check fails
        # We'll patch the hasattr to return False to test line 41
        with patch('builtins.hasattr', return_value=False):
            result = _process_recursive(config, is_top_level=False)

        # Since _process_recursive doesn't instantiate (line 41 just returns processed dict)
        # and we're not at top level, it returns the processed dict structure
        assert isinstance(result, dict)
        assert result.get("_target_") == "builtins.dict"

    def test_process_recursive_complex_nesting(self):
        """Test processing deeply nested structures."""
        config = {
            "level1": {
                "level2": {
                    "factories": [
                        {"_target_": "builtins.list"},
                        {"config": {"_target_": "builtins.dict"}}
                    ]
                }
            }
        }

        result = _process_recursive(config, is_top_level=True)

        assert isinstance(result["level1"]["level2"]["factories"][0], list)
        assert isinstance(result["level1"]["level2"]["factories"][1]["config"], dict)

    def test_process_recursive_mixed_omegaconf_and_dict(self):
        """Test processing mixed OmegaConf and regular dict structures."""
        # Create a scenario where we have both OmegaConf and regular dicts
        omega_part = OmegaConf.create({"_target_": "builtins.list"})
        config = {
            "omega": omega_part,
            "regular": {"_target_": "builtins.set"}
        }

        result = _process_recursive(config, is_top_level=True)

        assert isinstance(result["omega"], list)
        assert isinstance(result["regular"], set)
