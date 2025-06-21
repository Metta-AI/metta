"""
Unit tests for the typing library with BaseConfig and validation decorators.
"""

import os
import tempfile
from typing import List, Optional
from unittest.mock import patch

import pytest
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from metta.util.types.base_config import BaseConfig, ConfigRegistry, config_from_path
from metta.util.types.validator import (
    ValidatedConfig,
    auto_validate_config,
    extract_validated_configs,
    validate_config,
    validate_configs,
    validate_subconfig,
)


# Test configuration classes
class SimpleConfig(BaseConfig):
    """Simple test config."""

    __init__ = BaseConfig.__init__

    name: str
    value: int = 10


class NestedConfig(BaseConfig):
    """Config with nested fields."""

    __init__ = BaseConfig.__init__

    simple: SimpleConfig
    items: List[str]
    optional_value: Optional[float] = None


class StrictConfig(BaseConfig):
    """Config that forbids extra fields (default behavior)."""

    __init__ = BaseConfig.__init__

    required_field: str
    optional_field: int = 42


# Tests for BaseConfig
class TestBaseConfig:
    """Test cases for BaseConfig functionality."""

    def test_init_from_dict(self):
        """Test creating config from plain dict."""
        data = {"name": "test", "value": 20}
        config = SimpleConfig(data)
        assert config.name == "test"
        assert config.value == 20

    def test_init_from_dictconfig(self):
        """Test creating config from OmegaConf DictConfig."""
        data = OmegaConf.create({"name": "test", "value": 30})
        config = SimpleConfig(data)
        assert config.name == "test"
        assert config.value == 30

    def test_init_from_kwargs(self):
        """Test creating config using keyword arguments."""
        config = SimpleConfig(name="test", value=40)
        assert config.name == "test"
        assert config.value == 40

    def test_init_from_listconfig_raises_error(self):
        """Test that ListConfig input raises appropriate error."""
        list_cfg = OmegaConf.create([1, 2, 3])
        with pytest.raises(TypeError, match="Cannot create SimpleConfig from ListConfig"):
            SimpleConfig(list_cfg)

    def test_nested_config_with_listconfig_values(self):
        """Test that nested ListConfig values are properly converted to lists."""
        data = {"simple": {"name": "nested", "value": 5}, "items": ["a", "b", "c"]}
        cfg = OmegaConf.create(data)
        config = NestedConfig(cfg)
        assert isinstance(config.items, list)
        assert config.items == ["a", "b", "c"]

    def test_extra_fields_forbidden(self):
        """Test that extra fields raise validation error."""
        data = {"required_field": "test", "extra_field": "not allowed"}
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            StrictConfig(data)

    def test_dictconfig_conversion(self):
        """Test converting config back to DictConfig."""
        config = SimpleConfig(name="test", value=50)
        dict_cfg = config.dictconfig()
        assert isinstance(dict_cfg, DictConfig)
        assert dict_cfg.name == "test"
        assert dict_cfg.value == 50

    def test_yaml_serialization(self):
        """Test YAML output."""
        config = SimpleConfig(name="test", value=60)
        yaml_str = config.yaml()
        assert "name: test" in yaml_str
        assert "value: 60" in yaml_str

    def test_from_yaml(self):
        """Test loading config from YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("name: from_yaml\nvalue: 70\n")
            f.flush()

            try:
                config = SimpleConfig.from_yaml(f.name)
                assert config.name == "from_yaml"
                assert config.value == 70
            finally:
                os.unlink(f.name)

    def test_merge_with(self):
        """Test merging config with overrides."""
        config = SimpleConfig(name="original", value=80)
        overrides = {"value": 90}
        merged = config.merge_with(overrides)

        assert merged.name == "original"  # unchanged
        assert merged.value == 90  # overridden
        assert config.value == 80  # original unchanged

    def test_interpolation_resolution(self):
        """Test that OmegaConf interpolations are resolved."""
        data = OmegaConf.create({"base_value": 10, "name": "test", "value": "${base_value}"})
        config = SimpleConfig(data)
        assert config.value == 10

    def test_non_string_keys_error(self):
        """Test that non-string dict keys raise assertion error."""
        data = {1: "value", "name": "test"}
        with pytest.raises(AssertionError, match="All dictionary keys must be strings"):
            SimpleConfig(data)

    def test_prepare_dict_non_dict_error(self):
        """Test that non-dict data raises appropriate error."""
        config = SimpleConfig(name="test")
        with pytest.raises(TypeError, match="Data must be convertible to a dictionary"):
            config.prepare_dict("not a dict")


class TestConfigRegistry:
    """Test cases for ConfigRegistry."""

    def test_register_and_get(self):
        """Test registering and retrieving config classes."""
        registry = ConfigRegistry()
        registry.register("simple", SimpleConfig)

        assert registry.get("simple") == SimpleConfig
        assert registry.get("nonexistent") is None

    def test_register_path(self):
        """Test registering config for specific paths."""
        registry = ConfigRegistry()
        registry.register_path("configs/special", SimpleConfig)

        assert registry.get_by_path("configs/special") == SimpleConfig
        assert registry.get_by_path("configs/other") is None

    def test_validate(self):
        """Test validating config against registered type."""
        registry = ConfigRegistry()
        registry.register("simple", SimpleConfig)

        cfg = {"name": "test", "value": 100}
        validated = registry.validate("simple", cfg)

        assert isinstance(validated, SimpleConfig)
        assert validated.name == "test"
        assert validated.value == 100

    def test_validate_unregistered_error(self):
        """Test that validating unregistered group raises error."""
        registry = ConfigRegistry()
        cfg = {"name": "test"}

        with pytest.raises(KeyError, match="No config class registered for group: unknown"):
            registry.validate("unknown", cfg)


# Tests for validator decorators
class TestValidateConfig:
    """Test cases for validate_config decorator."""

    def test_validate_config_success(self):
        """Test successful config validation."""

        @validate_config(SimpleConfig)
        def process(cfg: SimpleConfig):
            return f"{cfg.name}:{cfg.value}"

        raw_cfg = OmegaConf.create({"name": "test", "value": 42})
        result = process(raw_cfg)
        assert result == "test:42"

    def test_validate_config_validation_error(self):
        """Test that validation errors are properly raised."""

        @validate_config(SimpleConfig)
        def process(cfg: SimpleConfig):
            return cfg.name

        raw_cfg = OmegaConf.create({"value": 42})  # missing required 'name'

        with pytest.raises(ValidationError):
            process(raw_cfg)

    def test_validate_config_listconfig_error(self):
        """Test that ListConfig input raises appropriate error."""

        @validate_config(SimpleConfig)
        def process(cfg: SimpleConfig):
            return cfg.name

        list_cfg = OmegaConf.create([1, 2, 3])

        with pytest.raises(TypeError, match="received ListConfig instead of DictConfig"):
            process(list_cfg)


class TestValidateSubconfig:
    """Test cases for validate_subconfig decorator."""

    def test_validate_subconfig_success(self):
        """Test successful subconfig validation."""

        @validate_subconfig("simple", SimpleConfig)
        def process(cfg: DictConfig):
            return cfg.simple.name

        raw_cfg = OmegaConf.create({"simple": {"name": "test", "value": 30}, "other": "data"})

        result = process(raw_cfg)
        assert result == "test"

    def test_validate_nested_subconfig(self):
        """Test validating nested field."""

        @validate_subconfig("level1.level2", SimpleConfig)
        def process(cfg: DictConfig):
            return cfg.level1.level2.name

        raw_cfg = OmegaConf.create({"level1": {"level2": {"name": "nested", "value": 40}}})

        result = process(raw_cfg)
        assert result == "nested"

    def test_validate_optional_missing_field(self):
        """Test that missing optional field doesn't raise error."""

        @validate_subconfig("optional", SimpleConfig, optional=True)
        def process(cfg: DictConfig):
            return "processed"

        raw_cfg = OmegaConf.create({"other": "data"})
        result = process(raw_cfg)
        assert result == "processed"

    def test_validate_required_missing_field(self):
        """Test that missing required field raises error."""

        @validate_subconfig("required", SimpleConfig, optional=False)
        def process(cfg: DictConfig):
            return "processed"

        raw_cfg = OmegaConf.create({"other": "data"})

        with pytest.raises(ValueError, match="Required config field 'required' not found"):
            process(raw_cfg)

    def test_validate_listconfig_in_path(self):
        """Test that ListConfig in path raises appropriate error."""

        @validate_subconfig("list_field.nested", SimpleConfig)
        def process(cfg: DictConfig):
            return "processed"

        raw_cfg = OmegaConf.create({"list_field": [1, 2, 3]})

        with pytest.raises(TypeError, match="Encountered ListConfig.*Cannot access dict keys in a list"):
            process(raw_cfg)

    def test_in_place_modification(self):
        """Test that in_place=True modifies the original config."""

        @validate_subconfig("simple", SimpleConfig, in_place=True)
        def process(cfg: DictConfig):
            # After validation, the field should be replaced with dict
            return isinstance(cfg.simple, dict)

        raw_cfg = OmegaConf.create({"simple": {"name": "test", "value": 50}})

        result = process(raw_cfg)
        assert result is True


class TestValidateConfigs:
    """Test cases for validate_configs decorator."""

    def test_validate_multiple_configs(self):
        """Test validating multiple config fields."""

        @validate_configs(
            ("config1", SimpleConfig, False),
            ("nested.config2", SimpleConfig, True),
        )
        def process(cfg: DictConfig):
            return f"{cfg.config1.name},{cfg.nested.get('config2', {}).get('name', 'missing')}"

        raw_cfg = OmegaConf.create(
            {"config1": {"name": "first", "value": 10}, "nested": {"config2": {"name": "second", "value": 20}}}
        )

        result = process(raw_cfg)
        assert result == "first,second"


class TestAutoValidateConfig:
    """Test cases for auto_validate_config decorator."""

    def test_auto_validate_registered_configs(self):
        """Test automatic validation of registered config types."""
        # Create a local registry for testing
        test_registry = ConfigRegistry()
        test_registry.register("simple", SimpleConfig)
        test_registry.register("nested", NestedConfig)

        @auto_validate_config
        def process(cfg: DictConfig):
            return cfg.simple.name

        # Mock the global registry
        with patch("metta.util.types.validator.config_registry", test_registry):
            raw_cfg = OmegaConf.create({"simple": {"name": "auto", "value": 60}, "unregistered": {"data": "ignored"}})

            result = process(raw_cfg)
            assert result == "auto"

    def test_auto_validate_listconfig_field(self):
        """Test that ListConfig fields are reported as errors."""
        test_registry = ConfigRegistry()
        test_registry.register("simple", SimpleConfig)

        @auto_validate_config
        def process(cfg: DictConfig):
            return "processed"

        with patch("metta.util.types.validator.config_registry", test_registry):
            raw_cfg = OmegaConf.create(
                {
                    "simple": [1, 2, 3]  # ListConfig instead of dict
                }
            )

            with pytest.raises(ValidationError, match="Field is a ListConfig"):
                process(raw_cfg)


class TestValidatedConfig:
    """Test cases for ValidatedConfig context manager."""

    def test_validated_config_success(self):
        """Test successful validation in context manager."""
        cfg = {"name": "context", "value": 70}

        with ValidatedConfig(cfg, SimpleConfig) as validated:
            assert isinstance(validated, SimpleConfig)
            assert validated.name == "context"
            assert validated.value == 70

    def test_validated_config_strict_mode_error(self):
        """Test that strict mode raises validation errors."""
        cfg = {"value": 80}  # missing required 'name'

        with pytest.raises(ValidationError):
            with ValidatedConfig(cfg, SimpleConfig, strict=True) as validated:
                pass

    def test_validated_config_non_strict_mode(self):
        """Test that non-strict mode returns best-effort instance."""
        cfg = {"value": 90}  # missing required 'name'

        with ValidatedConfig(cfg, SimpleConfig, strict=False) as validated:
            assert validated.value == 90
            # name will have its default or be unset

    def test_validated_config_listconfig_error(self):
        """Test that ListConfig input raises error."""
        list_cfg = OmegaConf.create([1, 2, 3])

        with pytest.raises(TypeError, match="received ListConfig instead of DictConfig"):
            with ValidatedConfig(list_cfg, SimpleConfig) as validated:
                pass


class TestExtractValidatedConfigs:
    """Test cases for extract_validated_configs function."""

    def test_extract_single_config(self):
        """Test extracting a single config."""
        cfg = OmegaConf.create({"simple": {"name": "extract", "value": 100}})

        configs = extract_validated_configs(cfg, simple=SimpleConfig)

        assert "simple" in configs
        assert isinstance(configs["simple"], SimpleConfig)
        assert configs["simple"].name == "extract"

    def test_extract_multiple_configs(self):
        """Test extracting multiple configs."""
        cfg = OmegaConf.create(
            {"config1": {"name": "first", "value": 10}, "nested": {"config2": {"name": "second", "value": 20}}}
        )

        configs = extract_validated_configs(cfg, config1=SimpleConfig, **{"nested.config2": SimpleConfig})

        assert len(configs) == 2
        assert configs["config1"].name == "first"
        assert configs["nested.config2"].name == "second"

    def test_extract_missing_field_error(self):
        """Test that missing fields raise appropriate error."""
        cfg = OmegaConf.create({"other": "data"})

        with pytest.raises(ValidationError, match="Failed to extract and validate configs"):
            extract_validated_configs(cfg, missing=SimpleConfig)

    def test_extract_listconfig_field_error(self):
        """Test that ListConfig fields raise appropriate error."""
        cfg = OmegaConf.create({"list_field": [1, 2, 3]})

        with pytest.raises(ValidationError, match="is a ListConfig"):
            extract_validated_configs(cfg, list_field=SimpleConfig)

    def test_extract_listconfig_root_error(self):
        """Test that ListConfig as root raises error."""
        list_cfg = OmegaConf.create([1, 2, 3])

        with pytest.raises(TypeError, match="received ListConfig instead of DictConfig"):
            extract_validated_configs(list_cfg, simple=SimpleConfig)


class TestConfigFromPath:
    """Test cases for config_from_path function."""

    @patch("hydra.compose")
    def test_config_from_path_success(self, mock_compose):
        """Test successful config loading from path."""
        mock_cfg = OmegaConf.create({"level1": {"config": {"name": "test", "value": 42}}})
        mock_compose.return_value = mock_cfg

        result = config_from_path("level1/config")

        assert result.name == "test"
        assert result.value == 42
        mock_compose.assert_called_once_with(config_name="level1/config")

    def test_config_from_path_none_error(self):
        """Test that None path raises error."""
        with pytest.raises(ValueError, match="Config path cannot be None"):
            config_from_path(None)

    def test_config_from_path_listconfig_overrides_error(self):
        """Test that ListConfig overrides raise error."""
        list_overrides = OmegaConf.create([1, 2, 3])

        with pytest.raises(TypeError, match="Overrides cannot be a ListConfig"):
            config_from_path("some/path", list_overrides)

    @patch("hydra.compose")
    def test_config_from_path_with_overrides(self, mock_compose):
        """Test config loading with overrides."""
        mock_cfg = OmegaConf.create({"config": {"name": "original", "value": 10}})
        mock_compose.return_value = mock_cfg

        overrides = {"value": 20}
        result = config_from_path("config", overrides)

        assert result.name == "original"
        assert result.value == 20

    @patch("hydra.compose")
    def test_config_from_path_listconfig_result_error(self, mock_compose):
        """Test that ListConfig result raises error."""
        mock_compose.return_value = OmegaConf.create([1, 2, 3])

        with pytest.raises(TypeError, match="is a ListConfig, expected DictConfig"):
            config_from_path("some/path")


# Integration tests
class TestIntegration:
    """Integration tests combining multiple features."""

    def test_nested_validation_workflow(self):
        """Test a complete workflow with nested configs."""

        # Define configs
        class DatabaseConfig(BaseConfig):
            __init__ = BaseConfig.__init__
            host: str
            port: int = 5432

        class AppConfig(BaseConfig):
            __init__ = BaseConfig.__init__
            name: str
            database: DatabaseConfig
            debug: bool = False

        # Create and validate
        raw_cfg = {"name": "MyApp", "database": {"host": "localhost", "port": 3306}, "debug": True}

        app_cfg = AppConfig(raw_cfg)
        assert app_cfg.name == "MyApp"
        assert app_cfg.database.host == "localhost"
        assert app_cfg.database.port == 3306
        assert app_cfg.debug is True

        # Test YAML serialization
        yaml_str = app_cfg.yaml()
        assert "name: MyApp" in yaml_str
        assert "host: localhost" in yaml_str

        # Test merging
        merged = app_cfg.merge_with({"database": {"port": 5432}})
        assert merged.database.port == 5432
        assert merged.database.host == "localhost"  # unchanged


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
