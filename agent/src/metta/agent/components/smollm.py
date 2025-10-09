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
    pad_value: int = 255

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
        self.pad_value = config.pad_value

        self._load_model()

        self.hidden_size = self.llm.config.hidden_size

        self.projector = nn.Linear(3, self.hidden_size)
        self.activation = nn.GELU()
        self.embed_norm = nn.LayerNorm(self.hidden_size)

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

        embeds, attention_mask = self._project_tokens(tokens)
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

        pooled = self._pool_hidden_states(final_hidden, attention_mask)
        pooled = pooled.to(dtype=self.actor_head.weight.dtype)

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
        self.projector = self.projector.to(device=device, dtype=llm_dtype)
        self.embed_norm = self.embed_norm.to(device=device, dtype=llm_dtype)
        self.activation = self.activation.to(device=device)
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
        attn_impl = self.config.attn_implementation
        dtype, attn_impl = self._harmonize_flash_attention(dtype, attn_impl)
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        if attn_impl is not None:
            kwargs["attn_implementation"] = attn_impl
        if attn_impl != self.config.attn_implementation:
            self.config.attn_implementation = attn_impl
        if dtype is not None and self.config.torch_dtype == "auto":
            self.config.torch_dtype = "bfloat16" if dtype == torch.bfloat16 else "float16"

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
        torch_dtype = self.config.torch_dtype
        if torch_dtype == "auto":
            return self._resolve_auto_dtype()
        return mapping[torch_dtype]

    def _resolve_auto_dtype(self) -> Optional[torch.dtype]:
        attn_impl = self.config.attn_implementation or ""
        if "flash_attention" in attn_impl:
            return self._preferred_flash_attention_dtype()
        return None

    def _preferred_flash_attention_dtype(self) -> Optional[torch.dtype]:
        if torch.cuda.is_available():
            is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)
            if is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        if torch.backends.mps.is_available():
            return torch.float16
        if torch.backends.mkldnn.is_available():
            return torch.bfloat16
        return None

    def _harmonize_flash_attention(
        self, dtype: Optional[torch.dtype], attn_impl: Optional[str]
    ) -> tuple[Optional[torch.dtype], Optional[str]]:
        if attn_impl is None or "flash_attention" not in attn_impl:
            return dtype, attn_impl
        if dtype in (torch.float16, torch.bfloat16):
            return dtype, attn_impl

        preferred = self._preferred_flash_attention_dtype() if dtype is None else None
        if preferred in (torch.float16, torch.bfloat16):
            logger.info("Setting FlashAttention dtype to %s", preferred)
            return preferred, attn_impl

        logger.warning(
            "Disabling FlashAttention because dtype %s is incompatible.",
            dtype if dtype is not None else "auto(float32)",
        )
        return dtype, None

    def _project_tokens(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.shape[-1] != 3:
            raise ValueError(f"Expected token tensors with last dim 3, got {tokens.shape[-1]}")

        batch_size, seq_len, _ = tokens.shape
        if seq_len == 0:
            tokens = torch.full(
                (batch_size, 1, 3),
                fill_value=self.pad_value,
                dtype=tokens.dtype,
                device=tokens.device,
            )
            seq_len = 1

        max_len = min(seq_len, self.max_sequence_length)
        tokens = tokens[:, :max_len]

        valid_mask = (tokens != self.pad_value).any(dim=-1)

        scaled_tokens = tokens.to(dtype=torch.float32) / 255.0
        scaled_tokens = torch.where(valid_mask.unsqueeze(-1), scaled_tokens, torch.zeros_like(scaled_tokens))
        scaled_tokens = scaled_tokens.to(dtype=self.projector.weight.dtype)

        embeds = self.projector(scaled_tokens)
        embeds = self.activation(embeds)
        embeds = self.embed_norm(embeds)

        attention_mask = valid_mask.to(dtype=torch.long)
        empty_rows = attention_mask.sum(dim=1) == 0
        if empty_rows.any():
            attention_mask = attention_mask.clone()
            attention_mask[empty_rows, 0] = 1

        return embeds, attention_mask

    def _pool_hidden_states(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        total = mask.sum(dim=1).clamp_min(1.0)
        return (hidden * mask).sum(dim=1) / total
