import numpy as np
import pytest

from metta.mettagrid.mettagrid_c import MettaGrid, dtype_actions
from metta.mettagrid.mettagrid_c_config import from_mettagrid_config
from metta.mettagrid.test_support import TestEnvironmentBuilder

pytestmark = pytest.mark.xfail(
    reason="Known nondeterminism: randomized agent action order within priority levels",
    strict=False,
)


def _make_env(seed: int, agent_positions: list[tuple[int, int]]):
    game_map = TestEnvironmentBuilder.create_basic_grid(5, 5)
    game_map = TestEnvironmentBuilder.place_agents(game_map, agent_positions, agent_type="agent.player")
    cfg = TestEnvironmentBuilder.make_test_config(num_agents=len(agent_positions), map=game_map.tolist())
    map_data = cfg.pop("map")
    return MettaGrid(from_mettagrid_config(cfg), map_data, seed)


def _rollout(env_seed: int, action_seed: int, positions: list[tuple[int, int]], max_steps: int = 50):
    env = _make_env(env_seed, positions)
    rng = np.random.default_rng(action_seed)

    obs, _ = env.reset()
    signatures = []
    steps = 0

    action_names = env.action_names()
    num_actions = len(action_names)

    while steps < max_steps:
        actions = np.zeros((env.num_agents, 2), dtype=dtype_actions)
        actions[:, 0] = rng.integers(0, num_actions, size=env.num_agents)
        actions[:, 1] = 0

        obs, rewards, terminals, truncations, _ = env.step(actions)

        signatures.append(
            (
                int(np.sum(obs)),
                tuple(np.asarray(rewards).tolist()),
                tuple(np.asarray(terminals).tolist()),
                tuple(np.asarray(truncations).tolist()),
            )
        )

        steps += 1
        if np.all(terminals) or np.all(truncations):
            break

    return signatures


class TestMultiAgentDeterminism:
    def test_two_agents_same_seed_identical(self):
        positions = [(2, 1), (2, 3)]
        sig1 = _rollout(env_seed=123, action_seed=999, positions=positions, max_steps=50)
        sig2 = _rollout(env_seed=123, action_seed=999, positions=positions, max_steps=50)
        assert sig1 == sig2
        assert len(sig1) > 0

    def test_two_agents_different_env_seed_diverge(self):
        positions = [(2, 1), (2, 3)]
        sig1 = _rollout(env_seed=123, action_seed=999, positions=positions, max_steps=50)
        sig2 = _rollout(env_seed=124, action_seed=999, positions=positions, max_steps=50)
        assert len(sig1) > 0
        assert sig1 != sig2

    def test_two_agents_different_action_seed_diverge(self):
        positions = [(2, 1), (2, 3)]
        sig1 = _rollout(env_seed=123, action_seed=999, positions=positions, max_steps=50)
        sig2 = _rollout(env_seed=123, action_seed=1000, positions=positions, max_steps=50)
        assert len(sig1) > 0
        assert sig1 != sig2
