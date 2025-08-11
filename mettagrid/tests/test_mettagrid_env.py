import pytest
from pydantic import ValidationError

from metta.mettagrid import AutoResetEnv


def test_invalid_env_config_type_raises():
    """Test that invalid env_config types raise proper errors."""
    with pytest.raises(ValidationError):
        # Passing a dict instead of EnvConfig should raise ValidationError
        AutoResetEnv(env_config={}, render_mode=None)


def test_invalid_env_config_missing_fields_raises():
    """Test that incomplete env_config raises proper errors."""
    with pytest.raises(ValidationError):
        # Passing an incomplete env_config should raise ValidationError
        AutoResetEnv(env_config={"game": {}}, render_mode=None)
