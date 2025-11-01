import torch
from tensordict import TensorDict

from metta.agent.policies.fast import FastConfig, FastPolicy
from metta.rl.utils import ensure_sequence_metadata
from mettagrid.config import MettaGridConfig
from mettagrid.policy.policy_env_interface import PolicyEnvInterface


def _build_policy_env_info() -> PolicyEnvInterface:
    return PolicyEnvInterface.from_mg_cfg(MettaGridConfig())


def _build_token_observations(batch_size: int, num_tokens: int) -> TensorDict:
    obs = torch.full((batch_size, num_tokens, 3), 0xFF, dtype=torch.uint8)
    # Token 0: coordinates (0,0), feature 0, value 10
    obs[:, 0] = torch.tensor([0x00, 0, 10], dtype=torch.uint8)
    # Token 1: coordinates (1,2) encoded as 0x12, feature 0, value 20
    obs[:, 1] = torch.tensor([0x12, 0, 20], dtype=torch.uint8)
    return TensorDict({"env_obs": obs}, batch_size=[batch_size])


def test_fast_config_creates_policy():
    policy_env_info = _build_policy_env_info()
    policy = FastConfig().make_policy(policy_env_info)
    assert isinstance(policy, FastPolicy)


def test_fast_policy_initialize_sets_action_metadata():
    policy_env_info = _build_policy_env_info()
    policy = FastPolicy(policy_env_info)

    logs = policy.initialize_to_environment(policy_env_info, torch.device("cpu"))

    # Initialization returns a list containing the observation shim log (may be None)
    assert isinstance(logs, list)
    assert policy.action_probs.num_actions == len(policy_env_info.actions.actions())


def test_fast_policy_forward_produces_actions_and_values():
    policy_env_info = _build_policy_env_info()
    policy = FastPolicy(policy_env_info)
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
