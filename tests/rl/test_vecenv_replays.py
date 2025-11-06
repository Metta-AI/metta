import metta.cogworks.curriculum
import metta.rl.vecenv
import metta.sim.replay_log_writer
import mettagrid.config.mettagrid_config


def test_make_env_func_creates_replay_writer(tmp_path):
    env_cfg = mettagrid.config.mettagrid_config.MettaGridConfig.EmptyRoom(num_agents=1)
    curriculum = metta.cogworks.curriculum.Curriculum(
        metta.cogworks.curriculum.CurriculumConfig(
            task_generator=metta.cogworks.curriculum.SingleTaskGenerator.Config(env=env_cfg)
        )
    )

    # Create a replay writer to pass in
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    replay_writer = metta.sim.replay_log_writer.ReplayLogWriter(str(replay_dir))

    env = metta.rl.vecenv.make_env_func(curriculum, replay_writer=replay_writer)
    try:
        # Verify that the simulation received the replay writer as an event handler
        # Note: With the new API, the replay writer is added to the simulation's event handlers
        assert any(
            isinstance(handler, metta.sim.replay_log_writer.ReplayLogWriter)
            for handler in env._env._simulator._event_handlers
        ), "ReplayLogWriter should be registered as an event handler"
    finally:
        env.close()
