import pytest
from omegaconf import OmegaConf

from mettagrid.mettagrid_env import MettaGridEnv


def test_invalid_env_map_type_raises():
    cfg = OmegaConf.create({})
    with pytest.raises(TypeError):
        MettaGridEnv(cfg, render_mode=None, env_map={})


def test_invalid_env_cfg_type_raises():
    with pytest.raises(TypeError):
        MettaGridEnv({}, render_mode=None)
