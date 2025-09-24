from types import SimpleNamespace

import gymnasium as gym
import torch
from tensordict import TensorDict

from metta.agent.policies.fast import FastConfig, FastPolicy
from metta.rl.training import EnvironmentMetaData
from metta.rl.utils import ensure_sequence_metadata


def _build_env_metadata():
    action_names = ["move", "attack"]
    max_action_args = [0, 2]  # move has no parameter, attack supports 3 arguments
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
    # Token 0: coordinates (0,0), feature 0, value 10
    obs[:, 0] = torch.tensor([0x00, 0, 10], dtype=torch.uint8)
    # Token 1: coordinates (1,2) encoded as 0x12, feature 0, value 20
    obs[:, 1] = torch.tensor([0x12, 0, 20], dtype=torch.uint8)
    return TensorDict({"env_obs": obs}, batch_size=[batch_size])


def test_fast_config_creates_policy():
    env_metadata = _build_env_metadata()
    policy = FastConfig().make_policy(env_metadata)
    assert isinstance(policy, FastPolicy)


def test_fast_policy_initialize_sets_action_metadata():
    env_metadata = _build_env_metadata()
    policy = FastPolicy(env_metadata)

    logs = policy.initialize_to_environment(env_metadata, torch.device("cpu"))

    # Initialization returns a list containing the observation shim log (may be None)
    assert isinstance(logs, list)
    assert policy.action_probs.action_index_tensor is not None
    assert policy.action_probs.cum_action_max_params is not None
    assert policy.action_probs.action_index_tensor.shape[1] == 2  # (action_type, action_param)


def test_fast_policy_forward_produces_actions_and_values():
    env_metadata = _build_env_metadata()
    policy = FastPolicy(env_metadata)
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
