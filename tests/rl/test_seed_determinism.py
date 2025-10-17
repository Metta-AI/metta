from types import SimpleNamespace
import random

import gymnasium as gym
import numpy as np
import torch

from metta.agent.policies.fast import FastConfig
from metta.rl.system_config import SystemConfig, seed_everything
from metta.rl.training.training_environment import GameRules


def _make_game_rules() -> GameRules:
    action_names = ["move", "attack"]
    feature_normalizations = {0: 1.0}
    obs_features = {
        "token_value": SimpleNamespace(id=0, normalization=1.0),
    }

    return GameRules(
        obs_width=32,
        obs_height=32,
        obs_features=obs_features,
        action_names=action_names,
        num_agents=1,
        observation_space=None,
        action_space=gym.spaces.Discrete(len(action_names)),
        feature_normalizations=feature_normalizations,
    )


def _build_policy(game_rules: GameRules) -> torch.nn.Module:
    policy_cfg = FastConfig()
    policy = policy_cfg.make_policy(game_rules)
    policy.initialize_to_environment(game_rules, torch.device("cpu"))
    return policy


def _clone_state_dict(policy: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}


def test_seed_reproducibility_for_policy_weights_and_rng_streams() -> None:
    system_cfg = SystemConfig(seed=1234, torch_deterministic=True, device="cpu")
    game_rules = _make_game_rules()

    def run_capture():
        seed_everything(system_cfg)
        policy = _build_policy(game_rules)
        weights = _clone_state_dict(policy)
        torch_sample = torch.randn(5, device="cpu")
        np_sample = np.random.rand(5)
        python_sample = [random.random() for _ in range(3)]
        return weights, torch_sample, np_sample, python_sample

    weights_a, torch_a, np_a, py_a = run_capture()

    # Mutate RNG streams to ensure the second capture truly depends on re-seeding.
    torch.rand(1)
    np.random.rand()
    random.random()

    weights_b, torch_b, np_b, py_b = run_capture()

    # Model parameters should be identical run-to-run with the same seed.
    for key, tensor_a in weights_a.items():
        tensor_b = weights_b[key]
        torch.testing.assert_close(tensor_a, tensor_b, msg=f"Parameter '{key}' differs between seeded runs")

    # RNG streams should also reproduce exactly.
    torch.testing.assert_close(torch_a, torch_b)
    np.testing.assert_allclose(np_a, np_b)
    assert py_a == py_b
