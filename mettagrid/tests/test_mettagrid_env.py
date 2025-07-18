import pytest
from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError
from pydantic import ValidationError

from metta.mettagrid.curriculum import single_task
from metta.mettagrid.mettagrid_env import MettaGridEnv


def test_invalid_env_map_type_raises():
    cfg = OmegaConf.create({})
    curriculum = single_task("test", cfg)
    with pytest.raises(ConfigAttributeError):
        MettaGridEnv(curriculum, render_mode=None, env_map={})


def test_invalid_env_cfg_type_raises():
    with pytest.raises(ValidationError):
        MettaGridEnv({}, render_mode=None)
