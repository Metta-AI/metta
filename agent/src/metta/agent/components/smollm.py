"""Components for policies backed by a pretrained SmolLLM model."""

from __future__ import annotations

import logging
from typing import Literal, Optional

import einops
import torch
import torch.nn.functional as F
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
    num_latents: int = 16
    perceiver_heads: int = 4
    perceiver_layers: int = 2
    coord_vocab_size: int = 256
    feature_vocab_size: int = 256

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
        self.coord_embed = nn.Embedding(self.config.coord_vocab_size, self.hidden_size)
        self.feature_embed = nn.Embedding(self.config.feature_vocab_size, self.hidden_size)
        self.value_embed = nn.Sequential(
            nn.Linear(1, self.hidden_size),
            nn.Tanh(),
        )

        self.token_norm = nn.LayerNorm(self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.num_latents = self.config.num_latents
        self.num_heads = self.config.perceiver_heads
        self.num_layers = self.config.perceiver_layers

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by perceiver_heads")

        self.latents = nn.Parameter(torch.randn(1, self.num_latents, self.hidden_size))
        nn.init.trunc_normal_(self.latents, std=0.02)

        self.perceiver_layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "latent_norm": nn.LayerNorm(self.hidden_size),
                        "q_proj": nn.Linear(self.hidden_size, self.hidden_size, bias=False),
                        "attn_out_proj": nn.Linear(self.hidden_size, self.hidden_size),
                        "mlp_norm": nn.LayerNorm(self.hidden_size),
                        "mlp": nn.Sequential(
                            nn.Linear(self.hidden_size, self.hidden_size * 4),
                            nn.GELU(),
                            nn.Linear(self.hidden_size * 4, self.hidden_size),
                        ),
                    }
                )
                for _ in range(self.num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(self.hidden_size)

        self.max_action_args = list(env.max_action_args)
        self.total_actions = sum(arg + 1 for arg in self.max_action_args)

        self.actor_head = pufferlib.pytorch.layer_init(nn.Linear(self.hidden_size, self.total_actions), std=0.01)
        self.value_head = pufferlib.pytorch.layer_init(nn.Linear(self.hidden_size, 1), std=1.0)

    def forward(self, td: TensorDict) -> TensorDict:
        flat_td = td.reshape(td.batch_size.numel()) if td.batch_dims > 1 else td
        tokens = flat_td[self.tokens_key]

        if tokens.dim() == 4:
            tokens = tokens.view(-1, tokens.shape[-2], tokens.shape[-1])

        tokens, mask = self._compress_tokens(tokens)
        token_features = self._embed_tokens(tokens, mask)
        latents = self._encode_latents(token_features, mask)

        llm_dtype = next(self.llm.parameters()).dtype
        latents = latents.to(dtype=llm_dtype)
        attention_mask = torch.ones(latents.shape[:2], dtype=torch.long, device=latents.device)

        outputs = self.llm(
            inputs_embeds=latents,
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
        self.coord_embed = self.coord_embed.to(device=device, dtype=llm_dtype)
        self.feature_embed = self.feature_embed.to(device=device, dtype=llm_dtype)
        self.value_embed = self.value_embed.to(device=device, dtype=llm_dtype)
        self.token_norm = self.token_norm.to(device=device, dtype=llm_dtype)
        self.k_proj = self.k_proj.to(device=device, dtype=llm_dtype)
        self.v_proj = self.v_proj.to(device=device, dtype=llm_dtype)
        self.latents = nn.Parameter(
            self.latents.to(device=device, dtype=llm_dtype), requires_grad=self.latents.requires_grad
        )
        self.perceiver_layers = self.perceiver_layers.to(device=device, dtype=llm_dtype)
        self.final_norm = self.final_norm.to(device=device, dtype=llm_dtype)
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

    def _embed_tokens(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        coords = tokens[..., 0].long()
        features = tokens[..., 1].long()
        values = tokens[..., 2].float().unsqueeze(-1)

        valid_mask = mask
        coords = torch.where(valid_mask, coords, torch.zeros_like(coords))
        features = torch.where(valid_mask, features, torch.zeros_like(features))
        values = torch.where(valid_mask.unsqueeze(-1), values, torch.zeros_like(values))

        coords = torch.clamp(coords, max=self.config.coord_vocab_size - 1)
        features = torch.clamp(features, max=self.config.feature_vocab_size - 1)

        coord_emb = self.coord_embed(coords)
        feat_emb = self.feature_embed(features)
        value_emb = self.value_embed(values)

        token_features = coord_emb + feat_emb + value_emb
        token_features = torch.where(valid_mask.unsqueeze(-1), token_features, torch.zeros_like(token_features))
        return token_features

    def _encode_latents(self, token_features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if token_features.numel() == 0:
            latents = self.latents.expand(token_features.shape[0], -1, -1)
            return latents

        tokens_norm = self.token_norm(token_features)
        k = self.k_proj(tokens_norm)
        v = self.v_proj(tokens_norm)

        k = einops.rearrange(k, "b m (h d) -> b h m d", h=self.num_heads)
        v = einops.rearrange(v, "b m (h d) -> b h m d", h=self.num_heads)

        attn_bias = None
        if not mask.all():
            mask_value = torch.finfo(k.dtype).min
            attn_bias = (~mask).unsqueeze(1).unsqueeze(2).to(k.dtype) * mask_value

        latents = self.latents.expand(token_features.shape[0], -1, -1)

        for layer in self.perceiver_layers:
            residual = latents
            q = layer["q_proj"](layer["latent_norm"](latents))
            q = einops.rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)

            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
            attn_output = einops.rearrange(attn_output, "b h n d -> b n (h d)")
            latents = residual + layer["attn_out_proj"](attn_output)

            latents = latents + layer["mlp"](layer["mlp_norm"](latents))

        latents = self.final_norm(latents)
        return latents
