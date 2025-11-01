import pytest
import torch
from tensordict import TensorDict

from metta.agent.policies import gtrxl as backbone_gtrxl
from metta.agent.policies import sliding_transformer as backbone_sliding
from metta.agent.policies import trxl as backbone_trxl
from metta.agent.policies import trxl_nvidia as backbone_trxl_nvidia
from metta.agent.policies.transformer import TransformerPolicy, TransformerPolicyConfig
from metta.rl.utils import ensure_sequence_metadata
from mettagrid.config import MettaGridConfig
from mettagrid.policy.policy_env_interface import PolicyEnvInterface


def _build_policy_env_info():
    return PolicyEnvInterface.from_mg_cfg(MettaGridConfig())


def _build_token_observations(batch_size: int, num_tokens: int) -> TensorDict:
    obs = torch.full((batch_size, num_tokens, 3), 0xFF, dtype=torch.uint8)
    obs[:, 0] = torch.tensor([0x00, 0, 10], dtype=torch.uint8)
    obs[:, 1] = torch.tensor([0x12, 0, 20], dtype=torch.uint8)
    return TensorDict({"env_obs": obs}, batch_size=[batch_size])


@pytest.mark.parametrize(
    ("config_factory", "expected_backbone"),
    [
        (backbone_gtrxl.gtrxl_policy_config, backbone_gtrxl.GTrXLConfig),
        (backbone_trxl.trxl_policy_config, backbone_trxl.TRXLConfig),
        (backbone_trxl_nvidia.trxl_nvidia_policy_config, backbone_trxl_nvidia.TRXLNvidiaConfig),
        (
            lambda: TransformerPolicyConfig(transformer=backbone_sliding.SlidingTransformerConfig()),
            backbone_sliding.SlidingTransformerConfig,
        ),
    ],
)
def test_transformer_config_creates_policy(config_factory, expected_backbone):
    policy_env_info = _build_policy_env_info()
    policy = config_factory().make_policy(policy_env_info)
    assert isinstance(policy, TransformerPolicy)
    assert isinstance(policy.config.transformer, expected_backbone)
    policy.initialize_to_environment(policy_env_info, torch.device("cpu"))
    policy.eval()

    td = _build_token_observations(batch_size=1, num_tokens=4)
    ensure_sequence_metadata(td, batch_size=1, time_steps=1)
    output_td = policy(td.clone())

    assert "actions" in output_td
    assert "values" in output_td
    assert output_td["actions"].shape == (1,)
    assert output_td["values"].shape == (1,)
    assert output_td["full_log_probs"].shape[0] == 1


def test_transformer_policy_initialization_sets_action_metadata():
    policy_env_info = _build_policy_env_info()
    policy = backbone_gtrxl.gtrxl_policy_config().make_policy(policy_env_info)

    policy.initialize_to_environment(policy_env_info, torch.device("cpu"))

    assert policy.action_probs.num_actions == len(policy_env_info.actions.actions())


def test_padding_tokens_do_not_zero_valid_entries():
    policy_env_info = _build_policy_env_info()
    policy = backbone_gtrxl.gtrxl_policy_config().make_policy(policy_env_info)
    policy.initialize_to_environment(policy_env_info, torch.device("cpu"))
    policy.eval()

    observations = torch.full((1, 4, 3), 0xFF, dtype=torch.uint8)
    observations[0, 0] = torch.tensor([0x00, 0, 10], dtype=torch.uint8)
    observations[0, 1] = torch.tensor([0xFF, 0, 0], dtype=torch.uint8)

    captured = {}

    def _capture_input(_, inputs):
        captured["grid"] = inputs[0].detach().clone()

    try:
        cnn1 = policy.cnn1
    except AttributeError:
        pytest.skip("Transformer policy does not expose a cnn1 module when using token encoders")

    handle = cnn1.register_forward_pre_hook(_capture_input)
    try:
        policy._encode_observations(observations)
    finally:
        handle.remove()

    assert "grid" in captured
    # Confirm that the valid token's value survives padding tokens (channel 0, location (0, 0)).
    assert captured["grid"][0, 0, 0, 0].item() == pytest.approx(10.0)


def test_transformer_reset_memory_is_noop():
    policy_env_info = _build_policy_env_info()
    policy = backbone_gtrxl.gtrxl_policy_config().make_policy(policy_env_info)
    policy.initialize_to_environment(policy_env_info, torch.device("cpu"))

    policy._memory[0] = torch.ones(1, policy.hidden_size)
    policy.reset_memory()

    assert 0 in policy._memory  # reset_memory should not clear transformer caches

    policy.clear_memory()
    assert policy._memory == {}


def test_transformer_memory_len_update():
    policy_env_info = _build_policy_env_info()
    policy = backbone_trxl.trxl_policy_config().make_policy(policy_env_info)
    policy.initialize_to_environment(policy_env_info, torch.device("cpu"))

    td = _build_token_observations(batch_size=1, num_tokens=4)
    ensure_sequence_metadata(td, batch_size=1, time_steps=1)
    policy(td.clone())

    original_len = policy.memory_len
    target_len = max(0, original_len // 2)
    policy.update_memory_len(target_len)
    assert policy.memory_len == target_len

    if original_len > 0:
        with pytest.raises(ValueError):
            policy.update_memory_len(original_len + 1)
