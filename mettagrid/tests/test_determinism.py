import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from metta.mettagrid.mettagrid_c import MettaGrid, dtype_actions
from metta.mettagrid.test_support import TestEnvironmentBuilder


class TestDeterminism:
    Signature = tuple[int, tuple[float, ...], tuple[bool, ...], tuple[bool, ...]]

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

    def test_seeded_uniform_policy_is_deterministic(self) -> None:
        sig1: list[TestDeterminism.Signature] = self._rollout(base_seed=1234, max_steps=30)
        sig2: list[TestDeterminism.Signature] = self._rollout(base_seed=1234, max_steps=30)
        assert sig1 == sig2

        sig3: list[TestDeterminism.Signature] = self._rollout(base_seed=4321, max_steps=30)
        assert len(sig1) > 0
        assert sig1 != sig3
