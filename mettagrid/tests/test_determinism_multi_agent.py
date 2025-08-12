from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from metta.mettagrid.mettagrid_c import MettaGrid, dtype_actions
from metta.mettagrid.mettagrid_c_config import from_mettagrid_config
from metta.mettagrid.test_support import TestEnvironmentBuilder

pytestmark = pytest.mark.xfail(
    reason="Known nondeterminism: randomized agent action order within priority levels",
    strict=False,
)


def _make_env(seed: int, agent_positions: Sequence[tuple[int, int]]) -> MettaGrid:
    game_map = TestEnvironmentBuilder.create_basic_grid(5, 5)
    game_map = TestEnvironmentBuilder.place_agents(game_map, list(agent_positions), agent_type="agent.player")
    cfg = TestEnvironmentBuilder.make_test_config(num_agents=len(agent_positions), map=game_map.tolist())
    map_data = cfg.pop("map")
    return MettaGrid(from_mettagrid_config(cfg), map_data, seed)


def _rollout(
    env_seed: int,
    action_seed: int,
    positions: Sequence[tuple[int, int]],
    max_steps: int = 50,
) -> list[tuple[int, tuple[float, ...], tuple[bool, ...], tuple[bool, ...]]]:
    env = _make_env(env_seed, positions)
    rng = np.random.default_rng(action_seed)

    obs, _ = env.reset()
    signatures: list[tuple[int, tuple[float, ...], tuple[bool, ...], tuple[bool, ...]]] = []
    steps: int = 0

    action_names: list[str] = env.action_names()
    num_actions: int = len(action_names)

    while steps < max_steps:
        actions: NDArray[np.int32] = np.zeros((env.num_agents, 2), dtype=dtype_actions)
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
    def test_two_agents_same_seed_identical(self) -> None:
        positions: list[tuple[int, int]] = [(2, 1), (2, 3)]
        sig1 = _rollout(env_seed=123, action_seed=999, positions=positions, max_steps=50)
        sig2 = _rollout(env_seed=123, action_seed=999, positions=positions, max_steps=50)
        assert sig1 == sig2
        assert len(sig1) > 0

    def test_two_agents_different_env_seed_diverge(self) -> None:
        positions: list[tuple[int, int]] = [(2, 1), (2, 3)]
        sig1 = _rollout(env_seed=123, action_seed=999, positions=positions, max_steps=50)
        sig2 = _rollout(env_seed=124, action_seed=999, positions=positions, max_steps=50)
        assert len(sig1) > 0
        assert sig1 != sig2

    def test_two_agents_different_action_seed_diverge(self) -> None:
        positions: list[tuple[int, int]] = [(2, 1), (2, 3)]
        sig1 = _rollout(env_seed=123, action_seed=999, positions=positions, max_steps=50)
        sig2 = _rollout(env_seed=123, action_seed=1000, positions=positions, max_steps=50)
        assert len(sig1) > 0
        assert sig1 != sig2
