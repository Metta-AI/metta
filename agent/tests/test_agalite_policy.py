from types import SimpleNamespace

import gymnasium as gym
import torch
from tensordict import TensorDict

from metta.agent.policies.agalite import AGaLiTeConfig
from metta.rl.training.training_environment import GameRules
from metta.rl.utils import ensure_sequence_metadata


def _build_game_rules():
    action_names = ["move", "attack"]
    feature_normalizations = {0: 1.0, 1: 10.0, 2: 5.0, 3: 2.0}

    obs_features = {
        "token_value": SimpleNamespace(id=0, normalization=1.0),
        "hp": SimpleNamespace(id=1, normalization=10.0),
        "shield": SimpleNamespace(id=2, normalization=5.0),
    }

    return GameRules(
        obs_width=11,
        obs_height=11,
        obs_features=obs_features,
        action_names=action_names,
        num_agents=1,
        observation_space=None,
        action_space=gym.spaces.Discrete(len(action_names)),
        feature_normalizations=feature_normalizations,
    )


def _build_token_observations(batch_size: int, num_tokens: int) -> TensorDict:
    obs = torch.full((batch_size, num_tokens, 3), 0xFF, dtype=torch.uint8)
    obs[:, 0] = torch.tensor([0x00, 0, 10], dtype=torch.uint8)
    obs[:, 1] = torch.tensor([0x12, 1, 20], dtype=torch.uint8)
    return TensorDict({"env_obs": obs}, batch_size=[batch_size])


def test_agalite_policy_forward():
    game_rules = _build_game_rules()
    policy_cfg = AGaLiTeConfig()
    policy = policy_cfg.make_policy(game_rules)

    policy.initialize_to_environment(game_rules, torch.device("cpu"))

    td = _build_token_observations(batch_size=1, num_tokens=4)
    ensure_sequence_metadata(td, batch_size=1, time_steps=1)

    output_td = policy(td.clone())

    assert "actions" in output_td
    assert "values" in output_td
    assert output_td["values"].shape == (1,)
