"""Tests for CLI configuration commands."""

import io
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from metta.config.schema import MettaConfig
from metta.setup.metta_cli import MettaCLI


class TestConfigureCLICommands:
    """Test metta configure CLI commands."""

    def test_export_env_command(self):
        """Test metta export-env command."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".metta" / "config.yaml"
            config_path.parent.mkdir(parents=True)

            # Create test config
            config = MettaConfig()
            config.wandb.enabled = True
            config.wandb.entity = "test-entity"
            config.observatory.enabled = False
            config.save(config_path)

            cli = MettaCLI()

            # Capture stdout
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()

            try:
                with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                    # Clear global config singleton to prevent test interference
                    import metta.config.schema as schema_module

                    schema_module._config = None
                    cli.cmd_export_env(Mock(), None)

                output = captured_output.getvalue()

                # Check expected output format
                assert "export WANDB_ENABLED='true'" in output
                assert "export WANDB_ENTITY='test-entity'" in output
                assert "export STATS_SERVER_ENABLED='false'" in output

            finally:
                sys.stdout = old_stdout

    def test_configure_component_unified(self):
        """Test configuring specific component."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".metta" / "config.yaml"
            config_path.parent.mkdir(parents=True)

            cli = MettaCLI()

            with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                # Clear global config singleton to prevent test interference
                import metta.config.schema as schema_module

                schema_module._config = None
                with patch("builtins.input", side_effect=["test-entity", "test-project", "y"]):
                    with (
                        patch("metta.setup.utils.header"),
                        patch("metta.setup.utils.success"),
                        patch("metta.setup.utils.info"),
                    ):
                        cli.configure_component_unified("wandb")

                # Verify config was saved
                assert config_path.exists()
                loaded_config = MettaConfig.load(config_path)
                assert loaded_config.wandb.entity == "test-entity"
                assert loaded_config.wandb.project == "test-project"
                assert loaded_config.wandb.enabled is True

    def test_configure_unknown_component(self):
        """Test error handling for unknown component."""
        cli = MettaCLI()

        with (
            patch("metta.setup.metta_cli.error") as mock_error,
            patch("metta.setup.metta_cli.info"),
            patch("sys.exit") as mock_exit,
        ):
            cli.configure_component_unified("nonexistent")

            mock_error.assert_called_once()
            mock_exit.assert_called_once_with(1)

    def test_configure_wizard(self):
        """Test interactive configuration wizard."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".metta" / "config.yaml"
            config_path.parent.mkdir(parents=True)

            cli = MettaCLI()

            # Simulate user selecting wandb and storage for configuration
            user_inputs = [
                "y",  # Should environment variables override config file values?
                "y",  # Configure Weights & Biases?
                "test-entity",  # W&B Entity
                "test-project",  # W&B Project
                "y",  # Enable W&B tracking?
                "y",  # Configure Storage?
                "my-bucket",  # S3 bucket name
                "y",  # Use S3 for replays?
                "n",  # Use S3 for torch traces?
                "n",  # Use S3 for checkpoints?
                "",  # AWS profile (blank)
                "n",  # Configure Observatory?
                "n",  # Configure Datadog?
            ]

            with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                # Clear global config singleton to prevent test interference
                import metta.config.schema as schema_module

                schema_module._config = None
                with patch("builtins.input", side_effect=user_inputs):
                    with (
                        patch("metta.setup.utils.header"),
                        patch("metta.setup.utils.success"),
                        patch("metta.setup.utils.info"),
                    ):
                        cli.configure_wizard()

                # Verify config was saved correctly
                assert config_path.exists()
                loaded_config = MettaConfig.load(config_path)
                assert loaded_config.wandb.entity == "test-entity"
                assert loaded_config.wandb.project == "test-project"
                assert loaded_config.wandb.enabled is True
                assert loaded_config.storage.s3_bucket == "my-bucket"
                assert loaded_config.storage.replay_dir == "s3://my-bucket/replays/"


class TestConfigHelperScript:
    """Test devops/config_helper.py script."""

    def test_export_env_format(self):
        """Test config_helper.py export --format=env."""
        from metta.devops.config_helper import export_env_vars

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".metta" / "config.yaml"
            config_path.parent.mkdir(parents=True)

            # Create test config
            config = MettaConfig()
            config.wandb.enabled = True
            config.wandb.entity = "helper-test"
            config.save(config_path)

            # Capture stdout
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()

            try:
                with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                    # Clear global config singleton to prevent test interference
                    import metta.config.schema as schema_module

                    schema_module._config = None
                    export_env_vars()

                output = captured_output.getvalue()
                assert "export WANDB_ENABLED='true'" in output
                assert "export WANDB_ENTITY='helper-test'" in output

            finally:
                sys.stdout = old_stdout

    def test_export_json_format(self):
        """Test config_helper.py export --format=json."""
        import json

        from metta.devops.config_helper import export_json

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".metta" / "config.yaml"
            config_path.parent.mkdir(parents=True)

            # Create test config
            config = MettaConfig()
            config.wandb.enabled = True
            config.wandb.entity = "json-test"
            config.storage.s3_bucket = "test-bucket"
            config.save(config_path)

            # Capture stdout
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()

            try:
                with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                    # Clear global config singleton to prevent test interference
                    import metta.config.schema as schema_module

                    schema_module._config = None
                    export_json()

                output = captured_output.getvalue()
                data = json.loads(output)

                assert data["wandb"]["enabled"] is True
                assert data["wandb"]["entity"] == "json-test"
                assert data["storage"]["s3_bucket"] == "test-bucket"
                assert data["profile"] == "external"

            finally:
                sys.stdout = old_stdout

    def test_export_file_format(self):
        """Test config_helper.py export --format=file."""
        from metta.devops.config_helper import export_env_file

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".metta" / "config.yaml"
            config_path.parent.mkdir(parents=True)
            env_file_path = Path(temp_dir) / ".env"

            # Create test config
            config = MettaConfig()
            config.wandb.enabled = True
            config.wandb.entity = "file-test"
            config.save(config_path)

            # Capture stdout for the success message
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()

            try:
                with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                    # Clear global config singleton to prevent test interference
                    import metta.config.schema as schema_module

                    schema_module._config = None
                    export_env_file(str(env_file_path))

                # Check success message
                output = captured_output.getvalue()
                assert "Configuration exported to" in output

                # Check .env file contents
                assert env_file_path.exists()
                with open(env_file_path) as f:
                    env_content = f.read()

                assert "WANDB_ENABLED=true" in env_content
                assert "WANDB_ENTITY=file-test" in env_content
                assert "STATS_SERVER_ENABLED=false" in env_content

            finally:
                sys.stdout = old_stdout

    def test_get_specific_value(self):
        """Test config_helper.py get command."""
        from metta.devops.config_helper import get_value

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".metta" / "config.yaml"
            config_path.parent.mkdir(parents=True)

            # Create test config
            config = MettaConfig()
            config.wandb.entity = "value-test"
            config.observatory.enabled = True
            config.save(config_path)

            # Test getting various values
            old_stdout = sys.stdout

            try:
                with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                    # Clear global config singleton to prevent test interference
                    import metta.config.schema as schema_module

                    schema_module._config = None
                    # Test string value
                    sys.stdout = captured_output = io.StringIO()
                    get_value("wandb.entity")
                    assert captured_output.getvalue().strip() == "value-test"

                    # Test boolean value
                    sys.stdout = captured_output = io.StringIO()
                    get_value("observatory.enabled")
                    assert captured_output.getvalue().strip() == "true"

                    # Test None value
                    sys.stdout = captured_output = io.StringIO()
                    get_value("wandb.project")
                    assert captured_output.getvalue().strip() == ""

            finally:
                sys.stdout = old_stdout
