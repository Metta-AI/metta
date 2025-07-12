import pytest
from pydantic import ValidationError

from metta.mettagrid.mettagrid_env import MettaGridEnv


@pytest.mark.hourly
def test_invalid_curriculum_type_raises():
    """Critical test for environment curriculum validation - runs hourly."""
    with pytest.raises(ValidationError):
        MettaGridEnv("not a curriculum", render_mode=None)


@pytest.mark.hourly
def test_invalid_render_mode_type_raises():
    """Critical test for render mode validation - runs hourly."""
    from omegaconf import OmegaConf

    from metta.mettagrid.curriculum.core import SingleTaskCurriculum

    cfg = OmegaConf.create({})
    curriculum = SingleTaskCurriculum("test", cfg)

    # render_mode should be str or None
    with pytest.raises(ValidationError):
        MettaGridEnv(curriculum, render_mode=123)  # invalid type


@pytest.mark.daily
def test_example_environment_initialization():
    """Test that example environments can be properly initialized - runs daily."""
    # This would test that get_example_env() works correctly
    # Placeholder for actual implementation
    pass
