from __future__ import annotations

from types import SimpleNamespace

import torch
from tensordict import TensorDict

from metta.agent.components.smollm import SmolLLMBackbone, SmolLLMBackboneConfig
from metta.agent.policies.smollm import SmolLLMConfig


class DummyOutput:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class DummyModel(torch.nn.Module):
    def __init__(self, hidden_size: int, dtype: torch.dtype):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.proj = torch.nn.Linear(hidden_size, hidden_size)
        self.proj.to(dtype=dtype)

    def forward(self, inputs_embeds, output_hidden_states, return_dict, attention_mask=None, use_cache=None):
        hidden = self.proj(inputs_embeds)
        return DummyOutput((inputs_embeds, hidden))


def _fake_from_pretrained(hidden_size: int = 64):
    def _factory(*args, **kwargs):
        dtype = kwargs.get("torch_dtype", torch.float32)
        if dtype is None or dtype == "auto":
            dtype = torch.float32
        return DummyModel(hidden_size=hidden_size, dtype=dtype)

    return _factory


def _make_env(max_action_args: list[int]) -> SimpleNamespace:
    return SimpleNamespace(max_action_args=max_action_args)


def _make_tokens(batch: int, seq: int) -> torch.Tensor:
    tokens = torch.full((batch, seq, 3), fill_value=255, dtype=torch.uint8)
    tokens[:, : seq // 2, :] = 1
    return tokens


def test_smollm_backbone_forward_sets_logits_and_values(monkeypatch):
    monkeypatch.setattr(
        "metta.agent.components.smollm.AutoModelForCausalLM",
        SimpleNamespace(from_pretrained=_fake_from_pretrained()),
    )

    env = _make_env([1, 2])
    config = SmolLLMBackboneConfig(
        in_key="tokens",
        logits_key="logits",
        values_key="values",
        max_sequence_length=8,
        freeze_llm=False,
        torch_dtype="float16",
    )
    backbone = SmolLLMBackbone(env, config)
    init_log = backbone.initialize_to_environment(env, torch.device("cpu"))
    assert "SmolLLM actions" in init_log

    td = TensorDict({"tokens": _make_tokens(2, 16)}, batch_size=[2])
    td_arg = td.clone()

    output = backbone(td_arg)
    assert output is td_arg
    assert td_arg["logits"].shape == (2, 5)
    assert td_arg["logits"].dtype == torch.float32
    assert td_arg["values"].shape == (2,)
    assert td_arg["values"].dtype == torch.float32


def test_smollm_config_builds_components():
    config = SmolLLMConfig(max_sequence_length=16, tokens_key="tokens")
    components = config.build_components()

    assert components[0].out_key == "tokens"
    assert components[1].logits_key == config.logits_key


def test_compress_tokens_vectorized(monkeypatch):
    monkeypatch.setattr(
        "metta.agent.components.smollm.AutoModelForCausalLM",
        SimpleNamespace(from_pretrained=_fake_from_pretrained()),
    )

    env = _make_env([1, 1])
    config = SmolLLMBackboneConfig(in_key="tokens", max_sequence_length=4)
    backbone = SmolLLMBackbone(env, config)

    tokens = torch.full((2, 6, 3), fill_value=255, dtype=torch.uint8)
    tokens[0, :3] = 1
    tokens[1, :2] = 1
    compressed, mask = backbone._compress_tokens(tokens)

    assert compressed.shape == (2, 3, 3)
    assert mask.shape == (2, 3)
    expected_mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]], dtype=torch.float32)
    torch.testing.assert_close(mask, expected_mask)
    # Compressed tokens should keep leading valid entries and zero-pad the rest
    assert torch.all(compressed[0, :3] != 0)
    assert torch.all(compressed[1, :2] != 0)
    assert torch.all(compressed[1, 2:] == 0)
