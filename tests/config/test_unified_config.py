"""Tests for the unified configuration system."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import yaml

from metta.config.components import CONFIGURATION_COMPONENTS
from metta.config.schema import MettaConfig, get_config, reload_config


class TestMettaConfig:
    """Test the unified configuration system."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MettaConfig()

        # Check defaults
        assert config.wandb.enabled is True
        assert config.wandb.entity is None
        assert config.wandb.project is None

        assert config.observatory.enabled is False
        assert config.observatory.stats_server_uri is None

        assert config.storage.s3_bucket is None
        assert config.storage.replay_dir is None

        assert config.profile == "external"

    def test_config_save_and_load(self):
        """Test saving and loading configuration to/from file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"

            # Create and save config
            config = MettaConfig()
            config.wandb.entity = "test-entity"
            config.wandb.project = "test-project"
            config.storage.s3_bucket = "test-bucket"
            config.profile = "cloud"

            config.save(config_path)

            # Verify file exists and has correct content
            assert config_path.exists()

            with open(config_path) as f:
                data = yaml.safe_load(f)

            assert data["wandb"]["entity"] == "test-entity"
            assert data["wandb"]["project"] == "test-project"
            assert data["storage"]["s3_bucket"] == "test-bucket"
            assert data["profile"] == "cloud"

            # Load config and verify
            loaded_config = MettaConfig.load(config_path)
            assert loaded_config.wandb.entity == "test-entity"
            assert loaded_config.wandb.project == "test-project"
            assert loaded_config.storage.s3_bucket == "test-bucket"
            assert loaded_config.profile == "cloud"

    def test_config_export_env_vars(self):
        """Test environment variable export."""
        config = MettaConfig()
        config.wandb.enabled = True
        config.wandb.entity = "my-team"
        config.wandb.project = "my-project"
        config.observatory.enabled = True
        config.observatory.stats_server_uri = "https://example.com/api"
        config.storage.aws_profile = "my-profile"
        config.storage.replay_dir = "s3://my-bucket/replays/"

        env_vars = config.export_env_vars()

        assert env_vars["WANDB_ENABLED"] == "true"
        assert env_vars["WANDB_ENTITY"] == "my-team"
        assert env_vars["WANDB_PROJECT"] == "my-project"
        assert env_vars["STATS_SERVER_ENABLED"] == "true"
        assert env_vars["STATS_SERVER_URI"] == "https://example.com/api"
        assert env_vars["AWS_PROFILE"] == "my-profile"
        assert env_vars["REPLAY_DIR"] == "s3://my-bucket/replays/"

    def test_config_empty_sections_omitted(self):
        """Test that empty config sections are omitted from saved file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"

            # Create config with only wandb settings
            config = MettaConfig()
            config.wandb.entity = "test-entity"
            # Leave other sections with default/None values

            config.save(config_path)

            with open(config_path) as f:
                data = yaml.safe_load(f)

            # Should only have wandb and profile sections
            assert "wandb" in data
            assert "profile" in data
            # Empty sections should be omitted
            assert "observatory" not in data
            assert "storage" not in data
            assert "datadog" not in data


class TestAutoConfigPriorityChain:
    """Test that auto_config functions respect the new priority chain."""

    def test_env_var_overrides_config_file(self):
        """Test environment variables override config file."""
        from metta.tools.utils.auto_config import auto_wandb_config

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".metta" / "config.yaml"
            config_path.parent.mkdir(parents=True)

            # Create config file with specific values
            config = MettaConfig()
            config.wandb.enabled = True
            config.wandb.entity = "config-entity"
            config.wandb.project = "config-project"
            config.save(config_path)

            # Mock home directory to use our temp config
            with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                # Clear global config singleton to prevent test interference
                import metta.config.schema as schema_module

                schema_module._config = None
                # Test without env vars - should use config file
                wandb_config = auto_wandb_config()
                assert wandb_config.entity == "config-entity"
                assert wandb_config.project == "config-project"

                # Test with env vars - should override config file
                with patch.dict(os.environ, {"WANDB_ENTITY": "env-entity", "WANDB_PROJECT": "env-project"}):
                    wandb_config = auto_wandb_config()
                    assert wandb_config.entity == "env-entity"
                    assert wandb_config.project == "env-project"

    def test_fallback_to_old_system(self):
        """Test fallback to old system when config file doesn't exist."""
        from metta.tools.utils.auto_config import auto_wandb_config

        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock home directory to non-existent config path
            with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                # Clear global config singleton to prevent test interference
                import metta.config.schema as schema_module

                schema_module._config = None
                # Should fall back to old system
                wandb_config = auto_wandb_config()
                # Old system should still work (returns defaults for external users)
                assert wandb_config.enabled is False  # External users default to disabled in old system
                assert wandb_config.entity == ""
                assert wandb_config.project == ""


class TestConfigurationComponents:
    """Test configuration components."""

    def test_all_components_registered(self):
        """Test that all expected components are registered."""
        expected_components = {"wandb", "storage", "observatory", "datadog"}
        actual_components = set(CONFIGURATION_COMPONENTS.keys())
        assert expected_components == actual_components

    def test_component_interfaces(self):
        """Test that all components implement required interface."""
        for name, component in CONFIGURATION_COMPONENTS.items():
            # Check required properties
            assert hasattr(component, "name")
            assert hasattr(component, "description")
            assert callable(component.interactive_configure)
            assert callable(component.apply_defaults)

            # Check they return expected types
            assert component.name == name
            assert isinstance(component.description, str)

    def test_apply_defaults_softmax_profile(self):
        """Test profile-based defaults for softmax users."""
        wandb_component = CONFIGURATION_COMPONENTS["wandb"]
        storage_component = CONFIGURATION_COMPONENTS["storage"]

        # Test softmax profile defaults
        wandb_config = wandb_component.apply_defaults({}, "softmax")
        storage_config = storage_component.apply_defaults({}, "softmax")

        assert wandb_config["enabled"] is True
        assert wandb_config["entity"] == "softmax-ai"
        assert "s3://softmax-public" in storage_config["replay_dir"]

    def test_apply_defaults_external_profile(self):
        """Test profile-based defaults for external users."""
        wandb_component = CONFIGURATION_COMPONENTS["wandb"]
        storage_component = CONFIGURATION_COMPONENTS["storage"]

        # Test external profile defaults
        wandb_config = wandb_component.apply_defaults({}, "external")
        storage_config = storage_component.apply_defaults({}, "external")

        assert wandb_config["enabled"] is False
        assert "./train_dir" in storage_config["replay_dir"]


class TestGlobalConfigInstance:
    """Test global config management."""

    def test_get_config_singleton(self):
        """Test that get_config returns singleton instance."""
        # Clear any existing instance
        import metta.config.schema as schema_module

        schema_module._config = None

        config1 = get_config()
        config2 = get_config()

        # Should be the same instance
        assert config1 is config2

    def test_reload_config(self):
        """Test config reloading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".metta" / "config.yaml"
            config_path.parent.mkdir(parents=True)

            # Create initial config
            config = MettaConfig()
            config.wandb.entity = "initial-entity"
            config.save(config_path)

            with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                # Clear global config singleton to prevent test interference
                import metta.config.schema as schema_module

                schema_module._config = None
                # Load initial config
                loaded_config = get_config()
                assert loaded_config.wandb.entity == "initial-entity"

                # Modify file on disk
                config.wandb.entity = "modified-entity"
                config.save(config_path)

                # get_config should return cached version
                cached_config = get_config()
                assert cached_config.wandb.entity == "initial-entity"

                # reload_config should pick up changes
                reloaded_config = reload_config()
                assert reloaded_config.wandb.entity == "modified-entity"
