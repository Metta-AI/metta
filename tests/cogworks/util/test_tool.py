"""Unit tests for cogworks.util.tool module."""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from pydantic import BaseModel

from cogworks.util.tool import run_tool


class SampleConfig(BaseModel):
    """Sample Pydantic config for testing."""

    name: str = "test"
    value: int = 42
    nested: dict = {"key": "value"}


def test_run_tool_with_pydantic_config():
    """Test that run_tool correctly handles Pydantic configs and saves with tabs."""
    config = SampleConfig(name="test_run", value=100)

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(spec=subprocess.CompletedProcess)

            # Run the tool
            run_tool("train", config, tmpdir)

            # Check that config was saved
            config_path = Path(tmpdir) / "tool_config.yaml"
            assert config_path.exists()

            # Read and verify the saved YAML content
            with open(config_path, "r") as f:
                yaml_content = f.read()

            # Verify the content is correct
            loaded_config = yaml.safe_load(yaml_content)
            assert loaded_config == config.model_dump()
            assert loaded_config["name"] == "test_run"
            assert loaded_config["value"] == 100
            assert loaded_config["nested"] == {"key": "value"}

            # Verify subprocess was called correctly
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert call_args[-1] == f"--config={config_path}"
            assert "train.py" in call_args[1]


def test_run_tool_with_non_pydantic_raises_error():
    """Test that passing a non-Pydantic object raises an AttributeError."""
    plain_dict = {"name": "test", "value": 42}

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(AttributeError, match="'dict' object has no attribute 'model_dump'"):
            run_tool("train", plain_dict, tmpdir)


def test_run_tool_creates_directory():
    """Test that run_tool creates the output directory if it doesn't exist."""
    config = SampleConfig()

    with tempfile.TemporaryDirectory() as tmpdir:
        nested_path = Path(tmpdir) / "nested" / "path"
        assert not nested_path.exists()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(spec=subprocess.CompletedProcess)
            run_tool("train", config, nested_path)

            assert nested_path.exists()
            assert (nested_path / "tool_config.yaml").exists()
