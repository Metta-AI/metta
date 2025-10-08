from __future__ import annotations

from types import SimpleNamespace

import torch
from gymnasium import spaces
from tensordict import TensorDict
import pytest

from metta.agent.components.smollm import SmolLLMBackbone, SmolLLMBackboneConfig


class DummyOutput:
    def __init__(self, hidden_states: tuple[torch.Tensor, ...]) -> None:
        self.hidden_states = hidden_states


class DummyModel(torch.nn.Module):
    def __init__(self, hidden_size: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.proj = torch.nn.Linear(hidden_size, hidden_size)
        self.proj.to(dtype=dtype)

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool,
        return_dict: bool,
        use_cache: bool,
    ) -> DummyOutput:
        del attention_mask, output_hidden_states, return_dict, use_cache
        hidden = self.proj(inputs_embeds)
        return DummyOutput((inputs_embeds, hidden))


def _fake_from_pretrained(hidden_size: int = 64):
    def _factory(*_: object, **kwargs: object) -> DummyModel:
        dtype = kwargs.get("torch_dtype", torch.float32)
        if not isinstance(dtype, torch.dtype):
            dtype = torch.float32
        return DummyModel(hidden_size=hidden_size, dtype=dtype)

    return _factory


def _make_env(num_actions: int) -> SimpleNamespace:
    return SimpleNamespace(action_space=spaces.Discrete(num_actions))


def _make_tokens(batch: int, seq: int, valid_prefix: int) -> torch.Tensor:
    tokens = torch.full((batch, seq, 3), fill_value=255, dtype=torch.uint8)
    tokens[:, :valid_prefix, 0] = 10
    tokens[:, :valid_prefix, 1] = 3
    tokens[:, :valid_prefix, 2] = 7
    return tokens


def test_backbone_forward_produces_float32_logits_and_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "metta.agent.components.smollm.AutoModelForCausalLM",
        SimpleNamespace(from_pretrained=_fake_from_pretrained()),
    )

    env = _make_env(num_actions=5)
    config = SmolLLMBackboneConfig(
        in_key="tokens",
        logits_key="logits",
        values_key="values",
        freeze_llm=False,
        torch_dtype="float16",
    )
    backbone = SmolLLMBackbone(env, config)
    init_log = backbone.initialize_to_environment(env, torch.device("cpu"))
    assert "SmolLLM actions" in init_log

    td = TensorDict({"tokens": _make_tokens(batch=2, seq=12, valid_prefix=6)}, batch_size=[2])
    td_copy = td.clone()

    output = backbone(td_copy)

    assert output is td_copy
    logits = td_copy["logits"]
    values = td_copy["values"]
    assert logits.shape == (2, 5)
    assert logits.dtype == torch.float32
    assert values.shape == (2,)
    assert values.dtype == torch.float32


def test_embed_tokens_builds_attention_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "metta.agent.components.smollm.AutoModelForCausalLM",
        SimpleNamespace(from_pretrained=_fake_from_pretrained()),
    )

    env = _make_env(num_actions=3)
    backbone = SmolLLMBackbone(env, SmolLLMBackboneConfig(in_key="tokens", max_sequence_length=4))

    tokens = _make_tokens(batch=2, seq=6, valid_prefix=3)
    embeds, attention_mask = backbone._embed_tokens(tokens)

    assert embeds.shape[:2] == attention_mask.shape
    assert embeds.shape[2] == backbone.hidden_size

    expected_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.long)
    assert torch.equal(attention_mask, expected_mask)


def test_embed_tokens_handles_all_padding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "metta.agent.components.smollm.AutoModelForCausalLM",
        SimpleNamespace(from_pretrained=_fake_from_pretrained()),
    )

    env = _make_env(num_actions=2)
    backbone = SmolLLMBackbone(env, SmolLLMBackboneConfig(in_key="tokens", max_sequence_length=2))

    tokens = torch.full((1, 5, 3), fill_value=255, dtype=torch.uint8)
    embeds, attention_mask = backbone._embed_tokens(tokens)

    assert embeds.shape == (1, 2, backbone.hidden_size)
    assert torch.equal(attention_mask, torch.tensor([[1, 0]], dtype=torch.long))
    assert torch.allclose(embeds, torch.zeros_like(embeds))
