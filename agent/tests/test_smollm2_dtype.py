"""Quick test to verify SmolLM2 dtype consistency."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict

from metta.rl.training import EnvironmentMetaData


class DummyFeature:
    def __init__(self, feature_id: int, normalization: float):
        self.id = feature_id
        self.normalization = normalization


class DummyLLM(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.proj = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, *, inputs_embeds, output_hidden_states: bool, return_dict: bool):
        hidden = self.proj(inputs_embeds)
        hidden_states = [inputs_embeds, hidden]
        if return_dict:
            return SimpleNamespace(hidden_states=hidden_states)
        return hidden_states


def _make_env_metadata(hidden_size: int = 576) -> EnvironmentMetaData:
    obs_features = {
        "feature_0": DummyFeature(0, 1.0),
        "feature_1": DummyFeature(1, 1.0),
        "feature_2": DummyFeature(2, 1.0),
    }
    feature_normalizations = {0: 1.0, 1: 1.0, 2: 1.0}
    action_names = ["move", "rotate", "noop"]
    max_action_args = [4, 2, 1]
    observation_space = SimpleNamespace(shape=(200, 3), dtype=torch.uint8)
    action_space = SimpleNamespace(nvec=torch.tensor([5, 3, 2], dtype=torch.int64))

    return EnvironmentMetaData(
        obs_width=16,
        obs_height=16,
        obs_features=obs_features,
        action_names=action_names,
        max_action_args=max_action_args,
        num_agents=1,
        observation_space=observation_space,
        action_space=action_space,
        feature_normalizations=feature_normalizations,
    )


def test_smollm2_dtype_consistency():
    """Test that SmolLM2 handles dtype mismatches correctly."""
    pytest.importorskip("transformers")
    from metta.agent.policies.smollm2 import SmolLM2Config, SmolLM2Policy

    env_metadata = _make_env_metadata()
    config = SmolLM2Config(model_name="dummy", freeze_llm=False)

    with patch("metta.agent.policies.smollm2.AutoModelForCausalLM.from_pretrained", return_value=DummyLLM(576)):
        policy = SmolLM2Policy(env_metadata, config)

    device = torch.device("cpu")
    policy.initialize_to_environment(env_metadata, device)

    batch_size = 2
    seq_len = 10
    observations = torch.randint(0, 255, (batch_size, seq_len, 3), dtype=torch.uint8)

    td = TensorDict({"env_obs": observations}, batch_size=[batch_size])
    with torch.no_grad():
        out_td = policy(td)

    assert "actions" in out_td
    assert "values" in out_td
    assert out_td["values"].dtype == torch.float32

    print("✅ SmolLM2 dtype conversion operates as expected")


if __name__ == "__main__":
    success = test_smollm2_dtype_consistency()
    exit(0 if success else 1)
