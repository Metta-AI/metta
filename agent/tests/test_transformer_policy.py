from types import SimpleNamespace

import gymnasium as gym
import torch
from tensordict import TensorDict

from metta.agent.policies.transformer import TransformerPolicy, TransformerPolicyConfig
from metta.rl.training.training_environment import EnvironmentMetaData
from metta.rl.utils import ensure_sequence_metadata


def _build_env_metadata():
    action_names = ["move", "attack"]
    max_action_args = [0, 2]
    feature_normalizations = {0: 1.0}

    obs_features = {
        "token_value": SimpleNamespace(id=0, normalization=1.0),
    }

    return EnvironmentMetaData(
        obs_width=11,
        obs_height=11,
        obs_features=obs_features,
        action_names=action_names,
        max_action_args=max_action_args,
        num_agents=1,
        observation_space=None,
        action_space=gym.spaces.MultiDiscrete([len(action_names), max(max_action_args) + 1]),
        feature_normalizations=feature_normalizations,
    )


def _build_token_observations(batch_size: int, num_tokens: int) -> TensorDict:
    obs = torch.full((batch_size, num_tokens, 3), 0xFF, dtype=torch.uint8)
    obs[:, 0] = torch.tensor([0x00, 0, 10], dtype=torch.uint8)
    obs[:, 1] = torch.tensor([0x12, 0, 20], dtype=torch.uint8)
    return TensorDict({"env_obs": obs}, batch_size=[batch_size])


def test_transformer_config_creates_policy():
    env_metadata = _build_env_metadata()
    policy = TransformerPolicyConfig().make_policy(env_metadata)
    assert isinstance(policy, TransformerPolicy)


def test_transformer_policy_forward_produces_actions_and_values():
    env_metadata = _build_env_metadata()
    policy = TransformerPolicy(env_metadata)
    policy.initialize_to_environment(env_metadata, torch.device("cpu"))
    policy.eval()

    td = _build_token_observations(batch_size=1, num_tokens=4)
    ensure_sequence_metadata(td, batch_size=1, time_steps=1)
    output_td = policy(td.clone())

    assert "actions" in output_td
    assert "values" in output_td
    assert output_td["actions"].shape == (1, 2)
    assert output_td["values"].shape == (1,)
    assert output_td["full_log_probs"].shape[0] == 1
