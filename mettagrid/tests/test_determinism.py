import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from metta.mettagrid.mettagrid_c import MettaGrid, dtype_actions
from metta.mettagrid.test_support import TestEnvironmentBuilder


class TestDeterminism:
    Signature = tuple[tuple[int, ...], bytes, tuple[float, ...], tuple[bool, ...], tuple[bool, ...]]

    def _rollout(self, base_seed: int, max_steps: int = 30) -> list[Signature]:
        builder: TestEnvironmentBuilder = TestEnvironmentBuilder()
        env: MettaGrid = builder.create_environment(num_agents=1)
        rng: Generator = np.random.default_rng(base_seed)

        obs, info = env.reset()
        signatures: list[TestDeterminism.Signature] = []
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
                    tuple(obs.shape),
                    obs.tobytes(),
                    tuple(np.asarray(rewards).tolist()),
                    tuple(np.asarray(terminals).tolist()),
                    tuple(np.asarray(truncations).tolist()),
                )
            )

            steps += 1
            if np.all(terminals) or np.all(truncations):
                break

        return signatures

    def test_seeded_uniform_policy_is_deterministic(self) -> None:
        sig1: list[TestDeterminism.Signature] = self._rollout(base_seed=1234, max_steps=30)
        sig2: list[TestDeterminism.Signature] = self._rollout(base_seed=1234, max_steps=30)
        assert sig1 == sig2

        sig3: list[TestDeterminism.Signature] = self._rollout(base_seed=4321, max_steps=30)
        assert len(sig1) > 0
        assert sig1 != sig3

    def _rollout_multi_agent(
        self, base_seed: int, positions: list[tuple[int, int]], max_steps: int = 30
    ) -> list[Signature]:
        builder: TestEnvironmentBuilder = TestEnvironmentBuilder()
        game_map = TestEnvironmentBuilder.create_basic_grid(width=25, height=25)
        game_map = TestEnvironmentBuilder.place_agents(game_map, positions, agent_type="agent.player")

        env: MettaGrid = builder.create_environment(game_map=game_map, num_agents=len(positions))
        rng: Generator = np.random.default_rng(base_seed)

        obs, _ = env.reset()
        signatures: list[TestDeterminism.Signature] = []
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
                    tuple(obs.shape),
                    obs.tobytes(),
                    tuple(np.asarray(rewards).tolist()),
                    tuple(np.asarray(terminals).tolist()),
                    tuple(np.asarray(truncations).tolist()),
                )
            )

            steps += 1
            if np.all(terminals) or np.all(truncations):
                break

        return signatures

    def test_seeded_uniform_policy_is_deterministic_multi_agent(self) -> None:
        positions: list[tuple[int, int]] = [
            (3, 3),
            (3, 12),
            (3, 21),
            (12, 3),
            (12, 12),
            (12, 21),
            (21, 3),
            (21, 12),
            (21, 21),
            (6, 18),
        ]

        sig1: list[TestDeterminism.Signature] = self._rollout_multi_agent(
            base_seed=1234, positions=positions, max_steps=30
        )
        sig2: list[TestDeterminism.Signature] = self._rollout_multi_agent(
            base_seed=1234, positions=positions, max_steps=30
        )
        assert sig1 == sig2

        sig3: list[TestDeterminism.Signature] = self._rollout_multi_agent(
            base_seed=4321, positions=positions, max_steps=30
        )
        assert len(sig1) > 0
        assert sig1 != sig3
