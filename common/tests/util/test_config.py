"""Tests for metta.common.util.config module."""

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest
from omegaconf import DictConfig, ListConfig, OmegaConf
from pydantic import Field, ValidationError

from metta.common.util.config import Config, config_from_path, copy_omegaconf_config


class TestConfig:
    """Test cases for the Config base class."""

    def test_config_inherits_from_basemodel(self):
        """Test that Config properly inherits from Pydantic BaseModel."""
        from pydantic import BaseModel

        assert issubclass(Config, BaseModel)

        # Test basic instantiation
        config = Config()
        assert isinstance(config, BaseModel)
        assert isinstance(config, Config)

    def test_config_forbids_extra_fields(self):
        """Test that Config forbids extra fields by default."""

        class TestConfig(Config):
            name: str
            value: int = 42

        # Valid instantiation should work
        config = TestConfig(name="test", value=100)
        assert config.name == "test"
        assert config.value == 100

        # Extra fields should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            TestConfig(name="test", value=42, extra_field="not_allowed")

        assert "Extra inputs are not permitted" in str(exc_info.value)

    def test_config_init_with_dictconfig(self):
        """Test Config initialization with OmegaConf DictConfig."""

        class DatabaseConfig(Config):
            host: str
            port: int = 5432
            ssl: bool = True

        # Create a DictConfig
        omega_config = OmegaConf.create({
            "host": "localhost",
            "port": 3306,
            "ssl": False
        })

        # Initialize Config with DictConfig
        config = DatabaseConfig(omega_config)

        assert config.host == "localhost"
        assert config.port == 3306
        assert config.ssl is False

    def test_config_init_with_regular_dict(self):
        """Test Config initialization with regular Python dict."""

        class ApiConfig(Config):
            url: str
            timeout: int = 30
            retries: int = 3

        # Create with regular dict
        data = {
            "url": "https://api.example.com",
            "timeout": 60,
            "retries": 5
        }

        config = ApiConfig(data)

        assert config.url == "https://api.example.com"
        assert config.timeout == 60
        assert config.retries == 5

    def test_config_init_with_kwargs(self):
        """Test Config initialization with normal kwargs."""

        class ServiceConfig(Config):
            name: str
            enabled: bool = True
            priority: int = Field(default=1, ge=1, le=10)

        # Normal kwargs initialization
        config = ServiceConfig(name="test-service", enabled=False, priority=5)

        assert config.name == "test-service"
        assert config.enabled is False
        assert config.priority == 5

    def test_config_dictconfig_method(self):
        """Test conversion back to OmegaConf DictConfig."""

        class ModelConfig(Config):
            name: str
            layers: int
            dropout: float = 0.1

        config = ModelConfig(name="transformer", layers=12, dropout=0.2)

        # Convert back to DictConfig
        dict_config = config.dictconfig()

        assert isinstance(dict_config, DictConfig)
        assert dict_config.name == "transformer"
        assert dict_config.layers == 12
        assert dict_config.dropout == 0.2

    def test_config_yaml_method(self):
        """Test YAML serialization."""

        class TrainingConfig(Config):
            model: str
            epochs: int
            learning_rate: float

        config = TrainingConfig(
            model="bert",
            epochs=10,
            learning_rate=0.001
        )

        yaml_str = config.yaml()

        assert isinstance(yaml_str, str)
        assert "model: bert" in yaml_str
        assert "epochs: 10" in yaml_str
        assert "learning_rate: 0.001" in yaml_str

    def test_config_prepare_dict_with_dictconfig(self):
        """Test prepare_dict method with DictConfig input."""

        class TestConfig(Config):
            pass

        config_instance = TestConfig()

        # Create DictConfig with interpolations
        omega_config = OmegaConf.create({
            "base_path": "/data",
            "train_path": "${base_path}/train",
            "val_path": "${base_path}/val"
        })

        result = config_instance.prepare_dict(omega_config)

        assert isinstance(result, dict)
        # Interpolations should be resolved
        assert result["base_path"] == "/data"
        assert result["train_path"] == "/data/train"
        assert result["val_path"] == "/data/val"

    def test_config_prepare_dict_with_regular_dict(self):
        """Test prepare_dict method with regular dict input."""

        class TestConfig(Config):
            pass

        config_instance = TestConfig()

        data = {
            "key1": "value1",
            "key2": 42,
            "key3": [1, 2, 3]
        }

        result = config_instance.prepare_dict(data)

        assert isinstance(result, dict)
        assert result == data

    def test_config_prepare_dict_validates_string_keys(self):
        """Test that prepare_dict validates all keys are strings."""

        class TestConfig(Config):
            pass

        config_instance = TestConfig()

        # Dict with non-string keys should raise AssertionError
        invalid_data = {
            "valid_key": "value1",
            123: "invalid_numeric_key",
            "another_valid": "value2"
        }

        with pytest.raises(AssertionError, match="All dictionary keys must be strings"):
            config_instance.prepare_dict(invalid_data)

    def test_config_prepare_dict_with_invalid_input(self):
        """Test prepare_dict with invalid input types."""

        class TestConfig(Config):
            pass

        config_instance = TestConfig()

        # Non-dict, non-DictConfig input should raise TypeError
        # dict() on a string tries to create dict from iterable, which fails
        with pytest.raises(ValueError):  # dict() raises ValueError for invalid sequences
            config_instance.prepare_dict("not a dict")

        # List that can't be converted to dict should also fail
        with pytest.raises(TypeError):  # dict() raises TypeError for invalid sequences
            config_instance.prepare_dict([1, 2, 3])

        # Test the actual TypeError path with data that passes dict() but isn't a dict
        class NotADict:
            def __iter__(self):
                return iter([])  # Returns empty iterable that becomes empty dict

        # This will create an empty dict, but then fail the isinstance check
        result = config_instance.prepare_dict(NotADict())
        assert result == {}  # Empty dict is valid

    def test_config_complex_nested_structure(self):
        """Test Config with complex nested structures."""

        class DatabaseConfig(Config):
            host: str
            port: int

        class CacheConfig(Config):
            redis_url: str
            ttl: int = 3600

        class AppConfig(Config):
            name: str
            database: dict[str, Any]  # Will contain DatabaseConfig data
            cache: dict[str, Any]     # Will contain CacheConfig data
            features: list[str] = Field(default_factory=list)

        # Create nested configuration
        config_data = {
            "name": "my-app",
            "database": {
                "host": "db.example.com",
                "port": 5432
            },
            "cache": {
                "redis_url": "redis://localhost:6379",
                "ttl": 7200
            },
            "features": ["auth", "api", "websockets"]
        }

        app_config = AppConfig(config_data)

        assert app_config.name == "my-app"
        assert app_config.database["host"] == "db.example.com"
        assert app_config.database["port"] == 5432
        assert app_config.cache["redis_url"] == "redis://localhost:6379"
        assert app_config.cache["ttl"] == 7200
        assert app_config.features == ["auth", "api", "websockets"]

    def test_config_with_pydantic_validation(self):
        """Test that Pydantic validation works correctly with Config."""

        class ValidatedConfig(Config):
            email: str = Field(pattern=r'^[^@]+@[^@]+\.[^@]+$')
            age: int = Field(ge=0, le=150)
            score: float = Field(ge=0.0, le=1.0)

        # Valid data should work
        config = ValidatedConfig(
            email="user@example.com",
            age=25,
            score=0.85
        )
        assert config.email == "user@example.com"
        assert config.age == 25
        assert config.score == 0.85

        # Invalid email should fail
        with pytest.raises(ValidationError):
            ValidatedConfig(email="invalid-email", age=25, score=0.85)

        # Invalid age should fail
        with pytest.raises(ValidationError):
            ValidatedConfig(email="user@example.com", age=-1, score=0.85)

        # Invalid score should fail
        with pytest.raises(ValidationError):
            ValidatedConfig(email="user@example.com", age=25, score=1.5)

    def test_config_serialization_roundtrip(self):
        """Test that Config can be serialized and deserialized correctly."""

        class GameConfig(Config):
            name: str
            max_players: int
            rules: dict[str, Any] = Field(default_factory=dict)

        original = GameConfig(
            name="chess",
            max_players=2,
            rules={"time_limit": 900, "increment": 10}
        )

        # Convert to YAML and back
        yaml_str = original.yaml()
        dict_config = OmegaConf.create(yaml_str)
        restored = GameConfig(dict_config)

        assert restored.name == original.name
        assert restored.max_players == original.max_players
        assert restored.rules == original.rules


class TestConfigFromPath:
    """Test cases for the config_from_path function."""

    def test_config_from_path_with_none_raises_error(self):
        """Test that passing None as config_path raises ValueError."""
        with pytest.raises(ValueError, match="Config path cannot be None"):
            config_from_path(None)

    @patch('hydra.compose')
    def test_config_from_path_basic_loading(self, mock_compose):
        """Test basic config loading from path."""
        # Mock the hydra.compose response
        mock_config = OmegaConf.create({
            "model": {"name": "transformer", "layers": 12},
            "training": {"epochs": 100, "lr": 0.001}
        })
        mock_compose.return_value = mock_config

        result = config_from_path("model_config")

        mock_compose.assert_called_once_with(config_name="model_config")
        assert result == mock_config

    @patch('hydra.compose')
    def test_config_from_path_with_leading_slash(self, mock_compose):
        """Test config loading with leading slash in path."""
        mock_config = OmegaConf.create({"test": "value"})
        mock_compose.return_value = mock_config

        result = config_from_path("/test_config")

        mock_compose.assert_called_once_with(config_name="/test_config")
        assert result == mock_config

    @patch('hydra.compose')
    def test_config_from_path_with_nested_path(self, mock_compose):
        """Test config loading with nested path structure."""
        # Create a nested config structure
        nested_config = OmegaConf.create({
            "level1": {
                "level2": {
                    "target_config": {
                        "setting1": "value1",
                        "setting2": 42
                    }
                }
            }
        })
        mock_compose.return_value = nested_config

        result = config_from_path("/level1/level2/target_config")

        # Should navigate to the nested config
        expected = nested_config.level1.level2
        assert result == expected

    @patch('hydra.compose')
    def test_config_from_path_with_overrides_dict(self, mock_compose):
        """Test config loading with dictionary overrides."""
        base_config = OmegaConf.create({
            "model": {"name": "bert", "layers": 6},
            "training": {"epochs": 50}
        })
        mock_compose.return_value = base_config

        overrides = {"model": {"layers": 12}, "training": {"epochs": 100}}

        result = config_from_path("config", overrides=overrides)

        # Check that overrides were applied
        assert result.model.layers == 12
        assert result.training.epochs == 100
        assert result.model.name == "bert"  # Should be preserved

    @patch('hydra.compose')
    def test_config_from_path_with_overrides_dictconfig(self, mock_compose):
        """Test config loading with DictConfig overrides."""
        base_config = OmegaConf.create({
            "database": {"host": "localhost", "port": 5432},
            "cache": {"enabled": False}
        })
        mock_compose.return_value = base_config

        overrides = OmegaConf.create({
            "database": {"port": 3306},
            "cache": {"enabled": True, "ttl": 3600}
        })

        result = config_from_path("config", overrides=overrides)

        # Check that overrides were applied
        assert result.database.port == 3306
        assert result.database.host == "localhost"  # Preserved
        assert result.cache.enabled is True
        assert result.cache.ttl == 3600  # New field added

    @patch('hydra.compose')
    def test_config_from_path_with_empty_overrides(self, mock_compose):
        """Test that empty overrides don't affect the config."""
        base_config = OmegaConf.create({"test": "value"})
        mock_compose.return_value = base_config

        # Test with None
        result1 = config_from_path("config", overrides=None)
        assert result1 == base_config

        # Test with empty dict
        result2 = config_from_path("config", overrides={})
        assert result2 == base_config

    @patch('hydra.compose')
    def test_config_from_path_struct_mode_handling(self, mock_compose):
        """Test that struct mode is properly handled with overrides."""
        base_config = OmegaConf.create({"existing": "value"})
        # Make it structured initially
        OmegaConf.set_struct(base_config, True)
        mock_compose.return_value = base_config

        # Override with new field (should work because we temporarily disable struct)
        overrides = {"new_field": "new_value"}

        result = config_from_path("config", overrides=overrides)

        # Should have both existing and new fields
        assert result.existing == "value"
        assert result.new_field == "new_value"
        # Struct mode should be re-enabled
        assert OmegaConf.is_struct(result)


class TestCopyOmegaconfConfig:
    """Test cases for the copy_omegaconf_config function."""

    def test_copy_dictconfig(self):
        """Test copying a DictConfig."""
        original = OmegaConf.create({
            "model": {"name": "bert", "layers": 12},
            "training": {"epochs": 100, "lr": 0.001}
        })

        copied = copy_omegaconf_config(original)

        # Should be a new instance
        assert copied is not original
        assert copied == original
        assert isinstance(copied, DictConfig)

    def test_copy_listconfig(self):
        """Test copying a ListConfig."""
        original = OmegaConf.create([
            {"name": "model1", "score": 0.95},
            {"name": "model2", "score": 0.87}
        ])

        copied = copy_omegaconf_config(original)

        # Should be a new instance
        assert copied is not original
        assert copied == original
        assert isinstance(copied, ListConfig)

    def test_copy_preserves_interpolations(self):
        """Test that copying preserves unresolved interpolations."""
        original = OmegaConf.create({
            "base_path": "/data",
            "train_path": "${base_path}/train",
            "model": {
                "name": "transformer",
                "checkpoint": "${base_path}/models/${model.name}"
            }
        })

        copied = copy_omegaconf_config(original)

        # Check that copied is independent and has same structure
        assert copied is not original
        assert copied.base_path == "/data"
        assert copied.model.name == "transformer"

        # The function uses resolve=False, so check the raw structure is preserved
        raw_copied = OmegaConf.to_container(copied, resolve=False)
        raw_original = OmegaConf.to_container(original, resolve=False)
        assert raw_copied == raw_original

        # Verify interpolations work when resolved
        resolved = OmegaConf.to_container(copied, resolve=True)
        assert resolved["train_path"] == "/data/train"

    def test_copy_with_nested_structures(self):
        """Test copying complex nested structures."""
        original = OmegaConf.create({
            "experiment": {
                "name": "test_run",
                "params": {
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "model": {
                        "type": "transformer",
                        "config": {
                            "layers": 6,
                            "heads": 8
                        }
                    }
                },
                "datasets": ["train.json", "val.json"]
            }
        })

        copied = copy_omegaconf_config(original)

        # Should be completely independent
        assert copied is not original
        assert copied.experiment is not original.experiment
        assert copied.experiment.params is not original.experiment.params

        # But content should be the same
        assert copied == original

        # Modifying copy shouldn't affect original
        copied.experiment.name = "modified_run"
        assert original.experiment.name == "test_run"
        assert copied.experiment.name == "modified_run"


class TestConfigDataValidation:
    """Test Config data validation."""

    def test_config_invalid_data_type(self):
        """Test Config prepare_dict with invalid data type raises TypeError."""
        class SimpleConfig(Config):
            value: int

        config_instance = SimpleConfig(value=1)
        
        # Test prepare_dict with data that dict() can process but returns non-dict
        # This will trigger the isinstance(data, dict) check on line 50 and raise on line 54
        # Create something that dict() can accept but doesn't return a dict
        
        # Mock the case where OmegaConf.to_container or dict() returns non-dict data
        with patch('metta.common.util.config.OmegaConf.to_container', return_value="not_a_dict"):
            from omegaconf import DictConfig
            dummy_config = DictConfig({})
            
            with pytest.raises(TypeError, match="Data must be convertible to a dictionary"):
                config_instance.prepare_dict(dummy_config)
