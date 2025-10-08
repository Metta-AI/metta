"""Components for policies backed by a pretrained SmolLLM model."""

from __future__ import annotations

import logging
from typing import Literal, Optional

import torch
from gymnasium.spaces import Discrete
from tensordict import TensorDict
from torch import nn

import pufferlib.pytorch
from metta.agent.components.component_config import ComponentConfig
from metta.rl.training import EnvironmentMetaData

logger = logging.getLogger(__name__)

try:
    from transformers import AutoModelForCausalLM
except ImportError as exc:  # pragma: no cover - import error path exercised in runtime, not tests
    AutoModelForCausalLM = None
    _IMPORT_ERROR = exc
else:  # pragma: no cover - exercised when dependency is installed
    _IMPORT_ERROR = None


class SmolLLMBackboneConfig(ComponentConfig):
    """Configuration for the SmolLLM backbone component."""

    in_key: str
    name: str = "smollm_backbone"
    logits_key: str = "smollm_logits"
    values_key: str = "values"
    hidden_key: Optional[str] = None

    model_name: str = "HuggingFaceTB/SmolLM2-135M"
    max_sequence_length: int = 32
    freeze_llm: bool = True
    torch_dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    attn_implementation: Optional[str] = "flash_attention_2"
    coord_vocab_size: int = 256
    feature_vocab_size: int = 256

    def make_component(self, env: EnvironmentMetaData):
        return SmolLLMBackbone(env, self)


class SmolLLMBackbone(nn.Module):
    """Backbone that projects Metta observation tokens into a pretrained SmolLLM."""

    def __init__(self, env: EnvironmentMetaData, config: SmolLLMBackboneConfig):
        super().__init__()

        if AutoModelForCausalLM is None:  # pragma: no cover - dependency missing in runtime
            raise ImportError("transformers is required to use SmolLLMBackbone") from _IMPORT_ERROR

        self.config = config
        self.tokens_key = config.in_key
        self.logits_key = config.logits_key
        self.values_key = config.values_key
        self.hidden_key = config.hidden_key
        self.max_sequence_length = config.max_sequence_length

        self._load_model()

        self.hidden_size = self.llm.config.hidden_size

        self.coord_embed = nn.Embedding(self.config.coord_vocab_size, self.hidden_size)
        self.feature_embed = nn.Embedding(self.config.feature_vocab_size, self.hidden_size)
        self.value_embed = nn.Sequential(nn.Linear(1, self.hidden_size), nn.Tanh())
        self.position_embed = nn.Embedding(self.max_sequence_length, self.hidden_size)
        self.token_norm = nn.LayerNorm(self.hidden_size)

        action_space = getattr(env, "action_space", None)
        if not isinstance(action_space, Discrete):
            raise TypeError(
                "SmolLLMBackbone requires a discrete action space; "
                f"received {type(action_space).__name__ if action_space is not None else 'None'}"
            )

        self.total_actions = int(action_space.n)

        self.actor_head = pufferlib.pytorch.layer_init(nn.Linear(self.hidden_size, self.total_actions), std=0.01)
        self.value_head = pufferlib.pytorch.layer_init(nn.Linear(self.hidden_size, 1), std=1.0)

    def forward(self, td: TensorDict) -> TensorDict:
        flat_td = td.reshape(td.batch_size.numel()) if td.batch_dims > 1 else td
        tokens = flat_td[self.tokens_key]

        if tokens.dim() == 4:
            tokens = tokens.view(-1, tokens.shape[-2], tokens.shape[-1])

        embeds, attention_mask = self._embed_tokens(tokens)
        llm_dtype = next(self.llm.parameters()).dtype
        embeds = embeds.to(dtype=llm_dtype)
        attention_mask = attention_mask.to(device=embeds.device)

        outputs = self.llm(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        final_hidden = outputs.hidden_states[-1]

        mask = attention_mask.unsqueeze(-1)
        pooled = (final_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        logits = self.actor_head(pooled).to(dtype=torch.float32)
        values = self.value_head(pooled).squeeze(-1).to(dtype=torch.float32)

        flat_td.set(self.logits_key, logits)
        flat_td.set(self.values_key, values)

        if self.hidden_key is not None:
            flat_td.set(self.hidden_key, pooled.to(dtype=torch.float32))

        return td

    def initialize_to_environment(self, env: EnvironmentMetaData, device: torch.device):
        self.to(device)

        llm_dtype = next(self.llm.parameters()).dtype
        self.coord_embed = self.coord_embed.to(device=device, dtype=llm_dtype)
        self.feature_embed = self.feature_embed.to(device=device, dtype=llm_dtype)
        self.value_embed = self.value_embed.to(device=device, dtype=llm_dtype)
        self.position_embed = self.position_embed.to(device=device, dtype=llm_dtype)
        self.token_norm = self.token_norm.to(device=device, dtype=llm_dtype)
        self.actor_head = self.actor_head.to(device=device, dtype=llm_dtype)
        self.value_head = self.value_head.to(device=device, dtype=llm_dtype)

        logger.info("SmolLLM backbone initialised with %d actions", self.total_actions)
        return f"SmolLLM actions: {self.total_actions}"

    def reset_memory(self):
        return None

    def _load_model(self) -> None:
        kwargs: dict[str, object] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        dtype = self._resolve_dtype()
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        if self.config.attn_implementation is not None:
            kwargs["attn_implementation"] = self.config.attn_implementation

        logger.info("Loading SmolLLM model '%s'", self.config.model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(self.config.model_name, **kwargs)

        if self.config.freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False

    def _resolve_dtype(self) -> Optional[torch.dtype]:
        mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if self.config.torch_dtype == "auto":
            return None
        return mapping[self.config.torch_dtype]

    def _embed_tokens(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.shape[-1] != 3:
            raise ValueError(f"Expected token tensors with last dim 3, got {tokens.shape[-1]}")

        batch_size = tokens.shape[0]
        max_len = min(tokens.shape[1], self.max_sequence_length)
        if max_len == 0:
            max_len = 1
            tokens = torch.zeros(batch_size, max_len, 3, dtype=tokens.dtype, device=tokens.device)
        else:
            tokens = tokens[:, :max_len]

        coord_tokens = tokens[..., 0].long()
        feature_tokens = tokens[..., 1].long()
        value_tokens = tokens[..., 2].float() / 255.0

        valid_mask = coord_tokens != 255

        coord_tokens = torch.clamp(coord_tokens, max=self.config.coord_vocab_size - 1)
        feature_tokens = torch.clamp(feature_tokens, max=self.config.feature_vocab_size - 1)

        zero_coords = torch.zeros_like(coord_tokens)
        zero_features = torch.zeros_like(feature_tokens)
        zero_values = torch.zeros_like(value_tokens)

        coord_tokens = torch.where(valid_mask, coord_tokens, zero_coords)
        feature_tokens = torch.where(valid_mask, feature_tokens, zero_features)
        value_tokens = torch.where(valid_mask, value_tokens, zero_values)

        coord_emb = self.coord_embed(coord_tokens)
        feature_emb = self.feature_embed(feature_tokens)
        value_tokens = value_tokens.to(dtype=self.value_embed[0].weight.dtype)
        value_emb = self.value_embed(value_tokens.unsqueeze(-1))

        token_features = coord_emb + feature_emb + value_emb

        position_ids = torch.arange(max_len, device=tokens.device).unsqueeze(0)
        position_emb = self.position_embed(position_ids)
        token_features = token_features + position_emb
        token_features = torch.where(valid_mask.unsqueeze(-1), token_features, torch.zeros_like(token_features))
        token_features = self.token_norm(token_features)

        attention_mask = valid_mask.to(dtype=torch.long)
        empty_rows = attention_mask.sum(dim=1) == 0
        if empty_rows.any():
            attention_mask = attention_mask.clone()
            attention_mask[empty_rows, 0] = 1
            token_features = token_features.clone()
            token_features[empty_rows, 0] = 0

        return token_features, attention_mask
