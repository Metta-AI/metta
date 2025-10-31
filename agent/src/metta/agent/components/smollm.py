"""Components for policies backed by a pretrained SmolLLM model."""

from __future__ import annotations

import logging
from typing import List, Literal, Optional, Protocol

import torch
from gymnasium.spaces import Discrete
from tensordict import TensorDict
from torch import nn

import pufferlib.pytorch
from metta.agent.components.component_config import ComponentConfig


class SupportsDiscreteActionSpace(Protocol):
    """Protocol describing the minimal environment interface required."""

    action_space: Discrete


try:  # pragma: no cover - optional training env module
    from metta.rl.training.training_environment import PolicyEnvInterface as _GameRules
except Exception:  # pragma: no cover - used when training package unavailable
    EnvInfo = SupportsDiscreteActionSpace
else:
    EnvInfo = _GameRules


logger = logging.getLogger(__name__)

try:
    from transformers import AutoModelForCausalLM
except Exception as exc:  # pragma: no cover - optional dependency may raise non-ImportError during import
    AutoModelForCausalLM = None
    _IMPORT_ERROR = exc
else:  # pragma: no cover - exercised when dependency is installed
    _IMPORT_ERROR = None

try:
    from peft import LoraConfig, TaskType, get_peft_model
except Exception:  # pragma: no cover - optional dependency may not be available
    LoraConfig = None
    TaskType = None
    get_peft_model = None


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
    token_stride: int = 1
    actor_head_rank: Optional[int] = None
    value_head_rank: Optional[int] = None
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None

    def make_component(self, env: EnvInfo) -> "SmolLLMBackbone":
        return SmolLLMBackbone(env, self)


class LowRankLinear(nn.Module):
    """Factorised linear layer to reduce parameters and activation memory."""

    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True) -> None:
        super().__init__()
        self.left = nn.Linear(in_features, rank, bias=bias)
        self.right = nn.Linear(rank, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.right(self.left(x))


class SmolLLMBackbone(nn.Module):
    """Backbone that projects Metta observation tokens into a pretrained SmolLLM."""

    def __init__(self, env: EnvInfo, config: SmolLLMBackboneConfig):
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
        self.token_stride = max(1, config.token_stride)

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

        self.actor_head = self._make_actor_head()
        self.value_head = self._make_value_head()

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
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states:
            final_hidden = hidden_states[-1]
        else:
            final_hidden = getattr(outputs, "last_hidden_state", None)
            if final_hidden is None:
                raise RuntimeError("SmolLLM backbone expected hidden states but model returned none")

        pooled = self._pool_hidden_states(final_hidden, attention_mask)
        pooled = pooled.to(dtype=self.actor_head.weight.dtype)

        logits = self.actor_head(pooled).to(dtype=torch.float32)
        values = self.value_head(pooled).squeeze(-1).to(dtype=torch.float32)

        flat_td.set(self.logits_key, logits)
        flat_td.set(self.values_key, values)

        if self.hidden_key is not None:
            flat_td.set(self.hidden_key, pooled.to(dtype=torch.float32))

        return td

    def initialize_to_environment(self, env: EnvInfo, device: torch.device):
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

        if self.config.use_lora:
            self._apply_lora()

        if self.config.freeze_llm:
            for name, param in self.llm.named_parameters():
                if self.config.use_lora and "lora_" in name:
                    param.requires_grad = True
                    continue
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

    def _apply_lora(self) -> None:
        if LoraConfig is None or get_peft_model is None or TaskType is None:
            raise ImportError("peft is required for LoRA support. Please install peft>=0.12.0")

        target_modules = self.config.lora_target_modules or [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ]

        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
        )

        logger.info(
            "Applying LoRA adapters (r=%d, alpha=%d, dropout=%.2f) to modules: %s",
            self.config.lora_rank,
            self.config.lora_alpha,
            self.config.lora_dropout,
            ", ".join(target_modules),
        )

        self.llm = get_peft_model(self.llm, lora_config)

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
        if self.token_stride > 1 and tokens.shape[1] > 1:
            tokens = tokens[:, :: self.token_stride]

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

    def _make_actor_head(self) -> nn.Module:
        rank = self.config.actor_head_rank
        if rank is None or rank >= min(self.hidden_size, self.total_actions):
            return pufferlib.pytorch.layer_init(nn.Linear(self.hidden_size, self.total_actions), std=0.01)
        layer = LowRankLinear(self.hidden_size, self.total_actions, rank)
        self._init_low_rank(layer, std=0.01)
        return layer

    def _make_value_head(self) -> nn.Module:
        rank = self.config.value_head_rank
        if rank is None or rank >= min(self.hidden_size, 1):
            return pufferlib.pytorch.layer_init(nn.Linear(self.hidden_size, 1), std=1.0)
        layer = LowRankLinear(self.hidden_size, 1, rank)
        self._init_low_rank(layer, std=1.0)
        return layer

    def _init_low_rank(self, layer: LowRankLinear, *, std: float) -> None:
        pufferlib.pytorch.layer_init(layer.left, std=std)
        pufferlib.pytorch.layer_init(layer.right, std=std)

    def _pool_hidden_states(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        total = mask.sum(dim=1).clamp_min(1.0)
        return (hidden * mask).sum(dim=1) / total
