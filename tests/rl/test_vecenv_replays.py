from pathlib import Path

from metta.cogworks.curriculum import Curriculum, CurriculumConfig, SingleTaskGenerator
from metta.rl.vecenv import make_env_func
from metta.sim.replay_log_renderer import ReplayLogRenderer
from mettagrid.config.mettagrid_config import MettaGridConfig


def test_make_env_func_creates_replay_writer(tmp_path):
    env_cfg = MettaGridConfig.EmptyRoom(num_agents=1)
    curriculum = Curriculum(CurriculumConfig(task_generator=SingleTaskGenerator.Config(env=env_cfg)))

    env = make_env_func(curriculum, replay_directory=str(tmp_path))
    try:
        assert isinstance(env._renderer, ReplayLogRenderer)
        renderer_path = Path(env._renderer._replay_dir)
        assert renderer_path.is_dir()
        assert str(renderer_path).startswith(str(tmp_path))
    finally:
        env.close()
