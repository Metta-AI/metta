import pytest
from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError
from pydantic_core import ValidationError

from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.mettagrid.mettagrid_config import EnvConfig


def test_invalid_env_map_type_raises():
    # Test that invalid env_map parameter raises ValidationError
    env_cfg = OmegaConf.create({})
    with pytest.raises(ValidationError):
        MettaGridEnv(env_config=env_cfg, render_mode=None)


def test_invalid_env_cfg_type_raises():
    with pytest.raises(ValidationError):
        MettaGridEnv({}, render_mode=None)
