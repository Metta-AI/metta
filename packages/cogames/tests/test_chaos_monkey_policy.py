from cogames.policy.chaos_monkey import ChaosMonkeyPolicy
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator.multi_episode.rollout import multi_episode_rollout


def test_chaos_monkey_rollout_records_failures() -> None:
    cfg = MettaGridConfig.EmptyRoom(num_agents=2, width=4, height=4, with_walls=True)
    cfg.game.max_steps = 6

    policy_env_info = PolicyEnvInterface.from_mg_cfg(cfg)
    policy = ChaosMonkeyPolicy(policy_env_info, fail_step=2, fail_probability=1.0)

    result = multi_episode_rollout(env_cfg=cfg, policies=[policy], episodes=1, seed=0)
    episode = result.episodes[0]

    assert episode.failure_steps is not None
    assert len(episode.failure_steps) == cfg.game.num_agents
    assert all(step == 2 for step in episode.failure_steps)
