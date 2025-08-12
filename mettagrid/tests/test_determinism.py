import numpy as np

from metta.mettagrid.mettagrid_c import dtype_actions
from metta.mettagrid.test_support import TestEnvironmentBuilder


class TestDeterminism:
    def _rollout(self, base_seed: int, max_steps: int = 30):
        builder = TestEnvironmentBuilder()
        env = builder.create_environment(num_agents=1)
        rng = np.random.default_rng(base_seed)

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

    def test_seeded_uniform_policy_is_deterministic(self):
        sig1 = self._rollout(base_seed=1234, max_steps=30)
        sig2 = self._rollout(base_seed=1234, max_steps=30)
        assert sig1 == sig2

        sig3 = self._rollout(base_seed=4321, max_steps=30)
        assert len(sig1) > 0
        assert sig1 != sig3
