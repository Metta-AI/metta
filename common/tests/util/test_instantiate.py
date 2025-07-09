"""
Unit tests for the instantiate utility function.
"""

import unittest

from omegaconf import DictConfig, OmegaConf

from metta.common.util.instantiate import instantiate


class SimpleClass:
    """A simple test class for instantiation."""

    def __init__(self, value: int, name: str = "default"):
        self.value = value
        self.name = name


class NestedClass:
    """A class that takes another object as a parameter."""

    def __init__(self, simple: SimpleClass, multiplier: int = 2):
        self.simple = simple
        self.multiplier = multiplier

    def get_result(self) -> int:
        return self.simple.value * self.multiplier


class ComplexClass:
    """A class with multiple nested components."""

    def __init__(self, components: list, metadata: dict):
        self.components = components
        self.metadata = metadata


class TestInstantiate(unittest.TestCase):
    """Test cases for the instantiate function."""

    def test_simple_instantiation(self):
        """Test basic instantiation with _target_ field."""
        config = {
            "_target_": f"{__name__}.SimpleClass",
            "value": 42,
            "name": "test",
        }
        obj = instantiate(config)
        self.assertIsInstance(obj, SimpleClass)
        self.assertEqual(obj.value, 42)
        self.assertEqual(obj.name, "test")

    def test_instantiation_with_dictconfig(self):
        """Test instantiation with OmegaConf DictConfig."""
        config = DictConfig(
            {
                "_target_": f"{__name__}.SimpleClass",
                "value": 10,
            }
        )
        obj = instantiate(config)
        self.assertIsInstance(obj, SimpleClass)
        self.assertEqual(obj.value, 10)
        self.assertEqual(obj.name, "default")  # Default value

    def test_instantiation_with_kwargs_override(self):
        """Test that kwargs override config values."""
        config = {
            "_target_": f"{__name__}.SimpleClass",
            "value": 10,
            "name": "config_name",
        }
        obj = instantiate(config, name="override_name")
        self.assertEqual(obj.value, 10)
        self.assertEqual(obj.name, "override_name")

    def test_recursive_instantiation(self):
        """Test recursive instantiation of nested configs."""
        config = {
            "_target_": f"{__name__}.NestedClass",
            "simple": {
                "_target_": f"{__name__}.SimpleClass",
                "value": 5,
                "name": "nested",
            },
            "multiplier": 3,
        }
        obj = instantiate(config, _recursive_=True)
        self.assertIsInstance(obj, NestedClass)
        self.assertIsInstance(obj.simple, SimpleClass)
        self.assertEqual(obj.simple.value, 5)
        self.assertEqual(obj.simple.name, "nested")
        self.assertEqual(obj.get_result(), 15)

    def test_recursive_with_list(self):
        """Test recursive instantiation with lists of configs."""
        config = {
            "_target_": f"{__name__}.ComplexClass",
            "components": [
                {
                    "_target_": f"{__name__}.SimpleClass",
                    "value": 1,
                    "name": "first",
                },
                {
                    "_target_": f"{__name__}.SimpleClass",
                    "value": 2,
                    "name": "second",
                },
            ],
            "metadata": {"key": "value"},
        }
        obj = instantiate(config, _recursive_=True)
        self.assertIsInstance(obj, ComplexClass)
        self.assertEqual(len(obj.components), 2)
        self.assertIsInstance(obj.components[0], SimpleClass)
        self.assertIsInstance(obj.components[1], SimpleClass)
        self.assertEqual(obj.components[0].value, 1)
        self.assertEqual(obj.components[1].value, 2)

    def test_missing_target_raises_error(self):
        """Test that missing _target_ field raises ValueError."""
        config = {"value": 42}
        with self.assertRaises(ValueError) as cm:
            instantiate(config)
        self.assertIn("missing '_target_'", str(cm.exception))

    def test_invalid_module_raises_error(self):
        """Test that invalid module path raises ImportError."""
        config = {"_target_": "nonexistent.module.Class"}
        with self.assertRaises(ImportError):
            instantiate(config)

    def test_invalid_class_raises_error(self):
        """Test that invalid class name raises AttributeError."""
        config = {"_target_": f"{__name__}.NonExistentClass"}
        with self.assertRaises(AttributeError):
            instantiate(config)

    def test_underscore_keys_excluded(self):
        """Test that underscore-prefixed keys are excluded from instantiation."""
        config = {
            "_target_": f"{__name__}.SimpleClass",
            "value": 10,
            "_extra_field": "should_be_ignored",
            "_another_field": 123,
        }
        obj = instantiate(config)
        self.assertEqual(obj.value, 10)
        # The underscore fields shouldn't cause any issues

    def test_non_recursive_does_not_instantiate_nested(self):
        """Test that non-recursive mode does not instantiate nested configs."""
        config = {
            "_target_": f"{__name__}.NestedClass",
            "simple": {
                "_target_": f"{__name__}.SimpleClass",
                "value": 5,
            },
            "multiplier": 3,
        }
        # Without _recursive_=True, the nested config is passed as a dict
        obj = instantiate(config, _recursive_=False)
        # The simple attribute should be a dict, not a SimpleClass instance
        self.assertIsInstance(obj.simple, dict)
        self.assertEqual(obj.simple["_target_"], f"{__name__}.SimpleClass")
        self.assertEqual(obj.simple["value"], 5)

    def test_omegaconf_resolution(self):
        """Test that OmegaConf interpolations are resolved."""
        config = OmegaConf.create(
            {
                "base_value": 10,
                "config": {
                    "_target_": f"{__name__}.SimpleClass",
                    "value": "${base_value}",
                    "name": "resolved",
                },
            }
        )
        obj = instantiate(config["config"])
        self.assertEqual(obj.value, 10)

    def test_invalid_config_type_raises_error(self):
        """Test that invalid config type raises TypeError."""
        with self.assertRaises(TypeError):
            instantiate("not_a_config")

        with self.assertRaises(TypeError):
            instantiate(123)


if __name__ == "__main__":
    unittest.main()
