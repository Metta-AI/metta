"""Unit tests for metta.common.util.typed_config module."""

import tempfile
from pathlib import Path
from typing import Optional

import pytest
import yaml
from pydantic import ValidationError

from metta.common.util.typed_config import ConfigWithBuilder, TypeSafeBuilder


class SampleConfig(ConfigWithBuilder):
    """Sample config class for testing."""

    name: str
    value: int
    enabled: bool = True
    optional_field: Optional[str] = None
    nested: dict = {}


class StrictConfig(ConfigWithBuilder):
    """Config that doesn't allow extra fields."""

    required_field: str
    number: int = 42


def test_builder_pattern():
    """Test the type-safe builder pattern."""
    config = SampleConfig.builder().name("test").value(100).enabled(False).build()

    assert config.name == "test"
    assert config.value == 100
    assert config.enabled is False
    assert config.optional_field is None
    assert config.nested == {}


def test_builder_with_partial_fields():
    """Test builder with only required fields set."""
    config = SampleConfig.builder().name("minimal").value(5).build()

    assert config.name == "minimal"
    assert config.value == 5
    assert config.enabled is True  # Default value
    assert config.optional_field is None
    assert config.nested == {}


def test_builder_with_all_fields():
    """Test builder with all fields including optional ones."""
    config = (
        SampleConfig.builder()
        .name("complete")
        .value(200)
        .enabled(True)
        .optional_field("extra")
        .nested({"key": "value"})
        .build()
    )

    assert config.name == "complete"
    assert config.value == 200
    assert config.enabled is True
    assert config.optional_field == "extra"
    assert config.nested == {"key": "value"}


def test_builder_missing_required_field():
    """Test that builder raises error when required field is missing."""
    with pytest.raises(ValidationError) as exc_info:
        SampleConfig.builder().name("test").build()  # Missing 'value'

    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("value",)
    assert errors[0]["type"] == "missing"


def test_from_file():
    """Test loading configuration from YAML file."""
    config_data = {
        "name": "from_file",
        "value": 300,
        "enabled": False,
        "optional_field": "loaded",
        "nested": {"loaded": True},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = Path(f.name)

    try:
        config = SampleConfig.from_file(temp_path)

        assert config.name == "from_file"
        assert config.value == 300
        assert config.enabled is False
        assert config.optional_field == "loaded"
        assert config.nested == {"loaded": True}
    finally:
        temp_path.unlink()


def test_from_file_with_string_path():
    """Test loading configuration from YAML file using string path."""
    config_data = {"name": "string_path", "value": 400}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name

    try:
        config = SampleConfig.from_file(temp_path)  # String path

        assert config.name == "string_path"
        assert config.value == 400
    finally:
        Path(temp_path).unlink()


def test_from_file_missing():
    """Test that from_file raises error for missing file."""
    with pytest.raises(FileNotFoundError) as exc_info:
        SampleConfig.from_file("/nonexistent/path/config.yaml")

    assert "Config file not found" in str(exc_info.value)


def test_from_file_invalid_yaml():
    """Test that from_file handles invalid YAML gracefully."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("invalid: yaml: content: [")  # Invalid YAML
        temp_path = Path(f.name)

    try:
        with pytest.raises(yaml.YAMLError):
            SampleConfig.from_file(temp_path)
    finally:
        temp_path.unlink()


def test_from_file_validation_error():
    """Test that from_file validates the loaded data."""
    config_data = {
        "name": "invalid",
        "value": "not_an_int",  # Wrong type
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = Path(f.name)

    try:
        with pytest.raises(ValidationError) as exc_info:
            SampleConfig.from_file(temp_path)

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("value",) for error in errors)
    finally:
        temp_path.unlink()


def test_extra_fields_forbidden():
    """Test that extra fields are forbidden by default."""
    with pytest.raises(ValidationError) as exc_info:
        StrictConfig(required_field="test", number=10, extra_field="not_allowed")

    errors = exc_info.value.errors()
    assert any("extra_field" in str(error) for error in errors)


def test_builder_chaining():
    """Test that builder methods can be chained."""
    builder = SampleConfig.builder()
    assert builder.name("test") is builder  # Same instance returned
    assert builder.value(100) is builder

    config = builder.build()
    assert config.name == "test"
    assert config.value == 100


def test_builder_overwrites_values():
    """Test that builder can overwrite previously set values."""
    config = (
        SampleConfig.builder()
        .name("first")
        .value(1)
        .name("second")  # Overwrite
        .value(2)  # Overwrite
        .build()
    )

    assert config.name == "second"
    assert config.value == 2


def test_type_safe_builder_initialization():
    """Test TypeSafeBuilder creates setters for all model fields."""
    builder = TypeSafeBuilder(SampleConfig)

    # Check that setters were created
    assert hasattr(builder, "name")
    assert hasattr(builder, "value")
    assert hasattr(builder, "enabled")
    assert hasattr(builder, "optional_field")
    assert hasattr(builder, "nested")

    # Check that they're callable
    assert callable(builder.name)
    assert callable(builder.value)
