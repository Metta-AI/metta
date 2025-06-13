import numpy as np
import pytest
from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError
from pydantic import ValidationError

from mettagrid.curriculum import SingleTaskCurriculum
from mettagrid.mettagrid_env import MettaGridEnv, dtype_actions
from mettagrid.room.random import Random


def test_invalid_env_map_type_raises():
    cfg = OmegaConf.create({})
    curriculum = SingleTaskCurriculum("test", cfg)
    with pytest.raises(ConfigAttributeError):
        MettaGridEnv(curriculum, render_mode=None, env_map={})


def test_invalid_env_cfg_type_raises():
    with pytest.raises(ValidationError):
        MettaGridEnv({}, render_mode=None)


def test_perf_metrics_in_infos():
    cfg = OmegaConf.load("mettagrid/tests/mettagrid_test_args_env_cfg.json")
    cfg.game.num_agents = 1
    cfg.game.max_steps = 1
    cfg.game.diversity_bonus = {"enabled": False}
    curriculum = SingleTaskCurriculum("test", cfg)
    level = Random(width=3, height=3, objects=OmegaConf.create({}), agents=1, border_width=1).build()
    env = MettaGridEnv(curriculum, render_mode=None, level=level)

    env.reset()
    noop_idx = env.action_names.index("noop")
    actions = np.array([[noop_idx, 0]], dtype=dtype_actions)
    _, _, _, _, infos = env.step(actions)

    expected = {"reset", "reset:make_c_env", "step", "step:episode_end"}
    assert isinstance(infos.get("perf"), dict)
    assert expected.issubset(infos["perf"].keys())
    for name in expected:
        assert isinstance(infos["perf"][name], float)
