from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch
from tensordict import TensorDict

from metta.agent.policies.transformer import (
    TransformerBackboneVariant,
    TransformerPolicy,
    gtrxl_policy_config,
    trxl_nvidia_policy_config,
    trxl_policy_config,
)
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


@pytest.mark.parametrize(
    ("config_factory", "variant"),
    [
        (gtrxl_policy_config, TransformerBackboneVariant.GTRXL),
        (trxl_policy_config, TransformerBackboneVariant.TRXL),
        (trxl_nvidia_policy_config, TransformerBackboneVariant.TRXL_NVIDIA),
    ],
)
def test_transformer_config_creates_policy(config_factory, variant):
    env_metadata = _build_env_metadata()
    policy = config_factory().make_policy(env_metadata)
    assert isinstance(policy, TransformerPolicy)
    assert policy.config.variant is variant
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


def test_transformer_policy_initialization_sets_action_metadata():
    env_metadata = _build_env_metadata()
    policy = gtrxl_policy_config().make_policy(env_metadata)

    policy.initialize_to_environment(env_metadata, torch.device("cpu"))

    assert policy.action_probs.action_index_tensor is not None
    assert policy.action_probs.cum_action_max_params is not None
    assert policy.action_index_tensor is policy.action_probs.action_index_tensor
    assert policy.cum_action_max_params is policy.action_probs.cum_action_max_params


def test_padding_tokens_do_not_zero_valid_entries():
    env_metadata = _build_env_metadata()
    policy = gtrxl_policy_config().make_policy(env_metadata)
    policy.initialize_to_environment(env_metadata, torch.device("cpu"))
    policy.eval()

    observations = torch.full((1, 4, 3), 0xFF, dtype=torch.uint8)
    observations[0, 0] = torch.tensor([0x00, 0, 10], dtype=torch.uint8)
    observations[0, 1] = torch.tensor([0xFF, 0, 0], dtype=torch.uint8)

    captured = {}

    def _capture_input(_, inputs):
        captured["grid"] = inputs[0].detach().clone()

    handle = policy.cnn1.register_forward_pre_hook(_capture_input)
    try:
        policy._encode_observations(observations)
    finally:
        handle.remove()

    assert "grid" in captured
    # Confirm that the valid token's value survives padding tokens (channel 0, location (0, 0)).
    assert captured["grid"][0, 0, 0, 0].item() == pytest.approx(10.0)
