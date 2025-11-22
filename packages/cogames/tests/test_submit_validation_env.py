from cogames.cli import submit


def test_load_validation_env_matches_mission_and_action_space() -> None:
    mission_name, env_cfg, env_info = submit._load_validation_env()

    assert mission_name == submit.DEFAULT_VALIDATION_MISSION
    assert env_info.action_space.n == len(env_cfg.game.actions.actions())
    assert env_info.action_space.n > 5  # ensure we're not validating on the EmptyRoom mock
