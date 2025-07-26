import pytest
from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError
from pydantic import ValidationError

from metta.mettagrid import MettaGridEnv
from metta.mettagrid.curriculum.core import SingleTaskCurriculum


def test_invalid_env_map_type_raises():
    cfg = OmegaConf.create({})
    curriculum = SingleTaskCurriculum("test", cfg)
    with pytest.raises(ConfigAttributeError):
        MettaGridEnv(curriculum, render_mode=None, env_map={})


def test_invalid_env_cfg_type_raises():
    with pytest.raises(ValidationError):
        MettaGridEnv({}, render_mode=None)
