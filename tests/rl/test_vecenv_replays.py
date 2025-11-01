from metta.cogworks.curriculum import Curriculum, CurriculumConfig, SingleTaskGenerator
from metta.rl.vecenv import make_env_func
from metta.sim.replay_log_writer import ReplayLogWriter
from mettagrid.config.mettagrid_config import MettaGridConfig


def test_make_env_func_creates_replay_writer(tmp_path):
    env_cfg = MettaGridConfig.EmptyRoom(num_agents=1)
    curriculum = Curriculum(CurriculumConfig(task_generator=SingleTaskGenerator.Config(env=env_cfg)))

    # Create a replay writer to pass in
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    replay_writer = ReplayLogWriter(str(replay_dir))

    env = make_env_func(curriculum, replay_writer=replay_writer)
    try:
        # Verify that the simulation received the replay writer as an event handler
        # Note: With the new API, the replay writer is added to the simulation's event handlers
        assert any(isinstance(handler, ReplayLogWriter) for handler in env._env._simulator._event_handlers), (
            "ReplayLogWriter should be registered as an event handler"
        )
    finally:
        env.close()
