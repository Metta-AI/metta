from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from gymnasium import spaces
from tensordict import TensorDict

from metta.agent.components.smollm import LowRankLinear, SmolLLMBackbone, SmolLLMBackboneConfig


class DummyOutput:
    def __init__(self, hidden_states: tuple[torch.Tensor, ...]) -> None:
        self.hidden_states = hidden_states


class RecordingModel(torch.nn.Module):
    def __init__(self, hidden_size: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.last_inputs_embeds: torch.Tensor | None = None
        self.last_attention_mask: torch.Tensor | None = None
        self.scale = torch.nn.Parameter(torch.tensor(2.0, dtype=dtype))

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool,
        return_dict: bool,
        use_cache: bool,
    ) -> DummyOutput:
        del output_hidden_states, return_dict, use_cache
        self.last_inputs_embeds = inputs_embeds
        self.last_attention_mask = attention_mask
        hidden = inputs_embeds * self.scale
        return DummyOutput((inputs_embeds, hidden))


def _fake_from_pretrained(hidden_size: int = 64):
    def _factory(*_: object, **kwargs: object) -> RecordingModel:
        dtype = kwargs.get("torch_dtype", torch.float32)
        if not isinstance(dtype, torch.dtype):
            dtype = torch.float32
        return RecordingModel(hidden_size=hidden_size, dtype=dtype)

    return _factory


def _capturing_from_pretrained(store: dict[str, object], hidden_size: int = 64):
    def _factory(*_: object, **kwargs: object) -> RecordingModel:
        store.update(kwargs)
        dtype = kwargs.get("torch_dtype", torch.float32)
        if not isinstance(dtype, torch.dtype):
            dtype = torch.float32
        return RecordingModel(hidden_size=hidden_size, dtype=dtype)

    return _factory


def _make_env(num_actions: int) -> SimpleNamespace:
    return SimpleNamespace(action_space=spaces.Discrete(num_actions))


def _make_tensordict(tokens: torch.Tensor) -> TensorDict:
    return TensorDict({"tokens": tokens}, batch_size=[tokens.shape[0]])


def test_forward_projects_tokens_and_calls_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "metta.agent.components.smollm.AutoModelForCausalLM",
        SimpleNamespace(from_pretrained=_fake_from_pretrained()),
    )

    env = _make_env(num_actions=4)
    config = SmolLLMBackboneConfig(
        in_key="tokens",
        logits_key="logits",
        values_key="values",
        max_sequence_length=4,
        freeze_llm=False,
        torch_dtype="float16",
    )
    backbone = SmolLLMBackbone(env, config)
    init_log = backbone.initialize_to_environment(env, torch.device("cpu"))
    assert "SmolLLM actions" in init_log

    tokens = torch.tensor(
        [
            [[1, 2, 3], [4, 5, 6], [255, 255, 255], [7, 8, 9], [10, 11, 12]],
            [[255, 255, 255], [2, 3, 4], [3, 4, 5], [255, 255, 255], [6, 7, 8]],
        ],
        dtype=torch.uint8,
    )
    td = _make_tensordict(tokens)

    output = backbone(td)

    assert output is td

    assert td["logits"].shape == (2, 4)
    assert td["logits"].dtype == torch.float32
    assert td["values"].shape == (2,)
    assert td["values"].dtype == torch.float32

    model = backbone.llm
    assert isinstance(model, RecordingModel)
    assert model.last_inputs_embeds is not None
    assert model.last_attention_mask is not None
    assert model.last_inputs_embeds.dtype == torch.float16
    expected_mask = torch.tensor([[1, 1, 0, 1], [0, 1, 1, 0]], dtype=torch.long)
    assert torch.equal(model.last_attention_mask, expected_mask)


def test_forward_adds_token_for_all_padding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "metta.agent.components.smollm.AutoModelForCausalLM",
        SimpleNamespace(from_pretrained=_fake_from_pretrained(hidden_size=32)),
    )

    env = _make_env(num_actions=3)
    backbone = SmolLLMBackbone(env, SmolLLMBackboneConfig(in_key="tokens", max_sequence_length=3, freeze_llm=False))
    backbone.initialize_to_environment(env, torch.device("cpu"))

    tokens = torch.full((1, 3, 3), fill_value=255, dtype=torch.uint8)
    td = _make_tensordict(tokens)
    backbone(td)

    model = backbone.llm
    assert isinstance(model, RecordingModel)
    assert model.last_inputs_embeds is not None
    assert model.last_attention_mask is not None
    assert model.last_inputs_embeds.shape[1] == 3
    expected_mask = torch.tensor([[1, 0, 0]], dtype=torch.long)
    assert torch.equal(model.last_attention_mask, expected_mask)


def test_token_stride_downsamples_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "metta.agent.components.smollm.AutoModelForCausalLM",
        SimpleNamespace(from_pretrained=_capturing_from_pretrained(captured, hidden_size=32)),
    )

    env = _make_env(num_actions=3)
    config = SmolLLMBackboneConfig(
        in_key="tokens",
        max_sequence_length=6,
        freeze_llm=False,
        torch_dtype="float16",
        token_stride=2,
    )
    backbone = SmolLLMBackbone(env, config)

    tokens = torch.tensor(
        [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]],
        ],
        dtype=torch.uint8,
    )
    td = _make_tensordict(tokens)
    backbone(td)

    model = backbone.llm
    assert isinstance(model, RecordingModel)
    assert model.last_inputs_embeds is not None
    assert model.last_inputs_embeds.shape[1] == 3


def test_lora_applies_and_keeps_adapter_grad(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class DummyLoraConfig:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    def _dummy_get_peft_model(model: RecordingModel, _: object) -> RecordingModel:
        model.lora_weight = torch.nn.Parameter(torch.ones(1, dtype=model.scale.dtype))
        return model

    monkeypatch.setattr(
        "metta.agent.components.smollm.AutoModelForCausalLM",
        SimpleNamespace(from_pretrained=_fake_from_pretrained(hidden_size=32)),
    )
    monkeypatch.setattr("metta.agent.components.smollm.LoraConfig", DummyLoraConfig)
    monkeypatch.setattr("metta.agent.components.smollm.get_peft_model", _dummy_get_peft_model)
    monkeypatch.setattr(
        "metta.agent.components.smollm.TaskType",
        SimpleNamespace(CAUSAL_LM="causal_lm"),
    )

    env = _make_env(num_actions=3)
    config = SmolLLMBackboneConfig(
        in_key="tokens",
        freeze_llm=True,
        torch_dtype="float16",
        use_lora=True,
        lora_rank=4,
        lora_alpha=32,
        lora_dropout=0.1,
        lora_target_modules=["foo", "bar"],
    )

    backbone = SmolLLMBackbone(env, config)

    assert isinstance(backbone.llm, RecordingModel)
    assert hasattr(backbone.llm, "lora_weight")
    assert not backbone.llm.scale.requires_grad
    assert backbone.llm.lora_weight.requires_grad
    assert captured["r"] == 4
    assert captured["target_modules"] == ["foo", "bar"]


def test_initialize_to_environment_aligns_module_dtypes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "metta.agent.components.smollm.AutoModelForCausalLM",
        SimpleNamespace(from_pretrained=_fake_from_pretrained(hidden_size=16)),
    )

    env = _make_env(num_actions=2)
    backbone = SmolLLMBackbone(
        env,
        SmolLLMBackboneConfig(in_key="tokens", freeze_llm=False, torch_dtype="float16"),
    )

    backbone.initialize_to_environment(env, torch.device("cpu"))

    assert backbone.projector.weight.dtype == torch.float16
    assert backbone.embed_norm.weight.dtype == torch.float16
    assert backbone.actor_head.weight.dtype == torch.float16
    assert backbone.value_head.weight.dtype == torch.float16


def test_flash_attn_auto_dtype_promotes_float16(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "metta.agent.components.smollm.AutoModelForCausalLM",
        SimpleNamespace(from_pretrained=_capturing_from_pretrained(captured, hidden_size=32)),
    )
    monkeypatch.setattr("metta.agent.components.smollm.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("metta.agent.components.smollm.torch.cuda.is_bf16_supported", lambda: False)
    monkeypatch.setattr("metta.agent.components.smollm.torch.backends.mps.is_available", lambda: False)
    monkeypatch.setattr("metta.agent.components.smollm.torch.backends.mkldnn.is_available", lambda: False)

    env = _make_env(num_actions=2)
    config = SmolLLMBackboneConfig(in_key="tokens", attn_implementation="flash_attention_2", freeze_llm=False)
    backbone = SmolLLMBackbone(env, config)
    assert isinstance(backbone.llm, RecordingModel)
    assert captured["torch_dtype"] == torch.float16
    assert captured["attn_implementation"] == "flash_attention_2"
    assert config.torch_dtype == "float16"
    assert config.attn_implementation == "flash_attention_2"


def test_flash_attn_disabled_when_no_supported_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "metta.agent.components.smollm.AutoModelForCausalLM",
        SimpleNamespace(from_pretrained=_capturing_from_pretrained(captured, hidden_size=32)),
    )
    monkeypatch.setattr("metta.agent.components.smollm.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("metta.agent.components.smollm.torch.backends.mps.is_available", lambda: False)
    monkeypatch.setattr("metta.agent.components.smollm.torch.backends.mkldnn.is_available", lambda: False)

    env = _make_env(num_actions=2)
    config = SmolLLMBackboneConfig(in_key="tokens", attn_implementation="flash_attention_2", freeze_llm=False)
    backbone = SmolLLMBackbone(env, config)
    assert isinstance(backbone.llm, RecordingModel)
    assert "attn_implementation" not in captured
    assert config.attn_implementation is None


def test_low_rank_actor_head(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "metta.agent.components.smollm.AutoModelForCausalLM",
        SimpleNamespace(from_pretrained=_fake_from_pretrained(hidden_size=64)),
    )

    env = _make_env(num_actions=20)
    config = SmolLLMBackboneConfig(
        in_key="tokens",
        freeze_llm=False,
        torch_dtype="float16",
        actor_head_rank=4,
        value_head_rank=1,
    )
    backbone = SmolLLMBackbone(env, config)

    assert isinstance(backbone.actor_head, LowRankLinear)
