import pytest
from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError
from pydantic_core import ValidationError

from metta.curriculum.rl.core import SingleTaskCurriculum
from metta.mettagrid.mettagrid_env import MettaGridEnv


def test_invalid_env_map_type_raises():
    env_cfg = OmegaConf.create({})
    curriculum = SingleTaskCurriculum("test", env_cfg)
    with pytest.raises(ValidationError):
        MettaGridEnv(env_cfg, curriculum=curriculum, render_mode=None, env_map={})


def test_invalid_env_cfg_type_raises():
    with pytest.raises(ValidationError):
        MettaGridEnv({}, render_mode=None)
