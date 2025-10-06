"""Components for policies backed by a pretrained SmolLLM model."""

from __future__ import annotations

import logging
from typing import Literal, Optional

import torch
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

    def make_component(self, env: EnvironmentMetaData):
        return SmolLLMBackbone(env, self)


class SmolLLMBackbone(nn.Module):
    """Backbone that projects Metta observations into a pretrained SmolLLM."""

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
        self.token_projector = nn.Sequential(
            nn.Linear(3, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self._initialize_projector()

        self.max_action_args = list(env.max_action_args)
        self.total_actions = sum(arg + 1 for arg in self.max_action_args)

        self.actor_head = pufferlib.pytorch.layer_init(nn.Linear(self.hidden_size, self.total_actions), std=0.01)
        self.value_head = pufferlib.pytorch.layer_init(nn.Linear(self.hidden_size, 1), std=1.0)

    def forward(self, td: TensorDict) -> TensorDict:
        flat_td = td.reshape(td.batch_size.numel()) if td.batch_dims > 1 else td
        tokens = flat_td[self.tokens_key]

        if tokens.dim() == 4:
            tokens = tokens.view(-1, tokens.shape[-2], tokens.shape[-1])

        tokens, attention_mask = self._compress_tokens(tokens)

        projector_dtype = next(self.token_projector.parameters()).dtype
        tokens = tokens.to(dtype=projector_dtype)
        embeddings = self.token_projector(tokens)

        llm_dtype = next(self.llm.parameters()).dtype
        embeddings = embeddings.to(dtype=llm_dtype)
        attention_mask = attention_mask.to(device=embeddings.device)

        outputs = self.llm(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        final_hidden = outputs.hidden_states[-1]
        pooled = final_hidden.mean(dim=1)

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
        self.token_projector = self.token_projector.to(device=device, dtype=llm_dtype)
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

    def _initialize_projector(self) -> None:
        for module in self.token_projector.modules():
            if isinstance(module, nn.Linear):
                pufferlib.pytorch.layer_init(module, std=1.0)

    def _resolve_dtype(self) -> Optional[torch.dtype]:
        mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if self.config.torch_dtype == "auto":
            return None
        return mapping[self.config.torch_dtype]

    def _compress_tokens(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Trim tokens to the effective max length and build an attention mask."""

        device = tokens.device
        valid_mask = tokens[..., 0] != 255
        lengths = valid_mask.sum(dim=1)
        if self.max_sequence_length is not None:
            lengths = torch.clamp(lengths, max=self.max_sequence_length)

        if lengths.numel() == 0:
            return tokens[:, :0], valid_mask[:, :0]

        max_len = int(torch.maximum(lengths.max(), torch.tensor(1, device=device)))
        if self.max_sequence_length is not None:
            max_len = min(max_len, self.max_sequence_length)

        gather_idx = torch.arange(max_len, device=device).expand(tokens.shape[0], -1)
        last_valid = torch.clamp(lengths - 1, min=0)
        gather_idx = torch.minimum(gather_idx, last_valid.unsqueeze(1)).to(torch.int64)

        gather_idx_expanded = gather_idx.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        gathered = torch.gather(tokens, 1, gather_idx_expanded)

        keep_mask = torch.arange(max_len, device=device).expand(tokens.shape[0], -1) < lengths.unsqueeze(1)
        fill = torch.full_like(gathered, 255)
        compressed = torch.where(keep_mask.unsqueeze(-1), gathered, fill)

        return compressed, keep_mask
