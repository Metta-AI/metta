from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from metta.agent.policies._transformer_utils import (
    _make_layer_norm,
    _record_function,
    empty_memory,
    normalize_memory,
    update_memory_window,
)


class FCPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with dropout."""

    def __init__(
        self,
        d_model: int,
        max_len: int = 8192,
        dropout: float = 0.1,
        *,
        scale_factor: float = 0.1,
    ) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", (pe * scale_factor).unsqueeze(1))  # (max_len, 1, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(0)
        if seq_len > self.pe.size(0):
            raise ValueError(f"Sequence length {seq_len} exceeds positional encoding capacity {self.pe.size(0)}.")
        return self.dropout(x + self.pe[:seq_len])


class FusedGRUGating(nn.Module):
    """Fused GRU-style gating used by GTrXL."""

    def __init__(self, d_model: int, bias: float = 2.0) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(2 * d_model, 3 * d_model, bias=False)
        self.bg = nn.Parameter(torch.full((d_model,), bias))
        nn.init.xavier_uniform_(self.gate_proj.weight, gain=1.0)

    def forward(self, residual: torch.Tensor, transformed: torch.Tensor) -> torch.Tensor:
        gates = self.gate_proj(torch.cat([residual, transformed], dim=-1))
        gates = gates.view(*gates.shape[:-1], 3, -1)
        reset = torch.sigmoid(gates[..., 0, :])
        update = torch.sigmoid(gates[..., 1, :] - self.bg)
        candidate = torch.tanh(gates[..., 2, :] * reset)
        return (1.0 - update) * residual + update * candidate


class GTrXLMultiHeadSelfAttention(nn.Module):
    """Multi-head attention with optional causal masking."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_causal_mask: bool = True,
        attn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_causal_mask = use_causal_mask

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self._attn_dropout_p = float(attn_dropout)
        self._dropout_p = float(dropout)

        nn.init.xavier_uniform_(self.qkv_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=1.0)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len, batch_size, _ = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.view(seq_len, batch_size, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 1, 3, 0, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q_2d = q.reshape(batch_size * self.n_heads, seq_len, self.d_k)
        k_2d = k.reshape(batch_size * self.n_heads, seq_len, self.d_k)
        v_2d = v.reshape(batch_size * self.n_heads, seq_len, self.d_k)

        scores = torch.bmm(q_2d, k_2d.transpose(1, 2)) / math.sqrt(self.d_k)

        if self.use_causal_mask:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))

        if attn_mask is not None:
            expanded = self._prepare_attn_mask(attn_mask, batch_size, seq_len, x.device)
            expanded = expanded.unsqueeze(1).expand(batch_size, self.n_heads, seq_len, seq_len)
            expanded = expanded.view(batch_size * self.n_heads, seq_len, seq_len)
            scores = scores.masked_fill(expanded, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=self._attn_dropout_p if self.training else 0.0, training=self.training)

        out = torch.bmm(weights, v_2d)
        out = out.view(batch_size, self.n_heads, seq_len, self.d_k)
        out = out.permute(2, 0, 1, 3).reshape(seq_len, batch_size, self.d_model)
        return self.out_proj(out)

    @staticmethod
    def _prepare_attn_mask(
        attn_mask: Optional[torch.Tensor],
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if attn_mask is None:
            return None

        mask = attn_mask
        if mask.dim() == 2:
            mask = mask.bool().unsqueeze(0).expand(batch_size, -1, -1)
        elif mask.dim() == 3:
            if mask.size(0) == 1:
                mask = mask.expand(batch_size, -1, -1)
            mask = mask.bool()
        else:
            raise ValueError("Attention mask must have dim 2 or 3.")

        return mask.to(device=device)


class GTrXLTransformerBlock(nn.Module):
    """GTrXL block with pre-layernorm and optional GRU-style gating."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        use_causal_mask: bool,
        use_gating: bool,
        *,
        attn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_gating = use_gating
        self.attention = GTrXLMultiHeadSelfAttention(
            d_model,
            n_heads,
            dropout,
            use_causal_mask,
            attn_dropout=attn_dropout,
        )
        self.norm1 = _make_layer_norm(d_model, False)
        self.norm2 = _make_layer_norm(d_model, False)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        for module in self.feed_forward:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)

        if use_gating:
            self.gate1 = FusedGRUGating(d_model, bias=2.0)
            self.gate2 = FusedGRUGating(d_model, bias=2.0)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attention(self.norm1(x), attn_mask)
        if self.use_gating:
            residual = self.gate1(x, attn_out)
        else:
            residual = x + attn_out

        ff_out = self.feed_forward(self.norm2(residual))
        if self.use_gating:
            return self.gate2(residual, ff_out)
        return residual + ff_out


class GTrXLModule(nn.Module):
    """GTrXL module matching the legacy full-context implementation."""

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 512,
        max_seq_len: int = 256,
        memory_len: int = 0,
        dropout: float = 0.1,
        use_gating: bool = True,
        use_causal_mask: bool = True,
        *,
        positional_scale: float = 0.1,
        attn_dropout: float = 0.1,
        activation_checkpoint: bool = False,
        use_flash_checkpoint: bool = False,
        use_fused_layernorm: bool = False,
        allow_tf32: bool = True,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.memory_len = max(0, memory_len)
        self.max_seq_len = max_seq_len
        self.use_input_proj = True
        self.use_gating = use_gating
        self.use_causal_mask = use_causal_mask
        self.use_activation_checkpoint = activation_checkpoint
        self.attn_dropout = attn_dropout
        self.use_flash_checkpoint = use_flash_checkpoint
        self.use_fused_layernorm = use_fused_layernorm
        self.allow_tf32 = allow_tf32
        if use_flash_checkpoint:
            warnings.warn(
                "GTrXLModule ignores use_flash_checkpoint; set this to False to silence the warning.",
                stacklevel=2,
            )
        if use_fused_layernorm:
            warnings.warn(
                "GTrXLModule ignores use_fused_layernorm; set this to False to silence the warning.",
                stacklevel=2,
            )

        positional_max = max_seq_len + self.memory_len + 1024
        self.positional_encoding = FCPositionalEncoding(
            d_model,
            max_len=positional_max,
            dropout=dropout,
            scale_factor=positional_scale,
        )
        if self.use_input_proj:
            self.input_proj = nn.Linear(d_model, d_model)
            nn.init.xavier_uniform_(self.input_proj.weight, gain=1.0)
            nn.init.constant_(self.input_proj.bias, 0.0)

        self.layers = nn.ModuleList(
            [
                GTrXLTransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    use_causal_mask=use_causal_mask,
                    use_gating=self.use_gating,
                    attn_dropout=attn_dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.output_norm = _make_layer_norm(d_model, False)
        self.dropout = nn.Dropout(dropout)
        self._mask_cache: dict[tuple[int, torch.device], torch.Tensor] = {}

    def forward(
        self,
        inputs: torch.Tensor,
        memory: Optional[Dict[str, Optional[List[torch.Tensor]]]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Optional[List[torch.Tensor]]]]:
        with _record_function("GTrXLModule/forward"):
            squeeze = False
            if inputs.dim() == 2:
                inputs = inputs.unsqueeze(0)
                squeeze = True
            if inputs.dim() != 3:
                raise ValueError(f"Expected tensor of shape (T, B, D); received {inputs.shape}.")

            _, batch_size, _ = inputs.shape
            device = inputs.device
            dtype = inputs.dtype

            with _record_function("GTrXLModule/normalize_memory"):
                stored_memory = memory.get("hidden_states") if isinstance(memory, dict) else None
                layer_mems = normalize_memory(
                    self.memory_len,
                    self.n_layers,
                    stored_memory,
                    batch_size,
                    self.d_model,
                    device,
                    dtype,
                )
                if layer_mems is None:
                    layer_mems = empty_memory(self.n_layers, batch_size, self.d_model, device, dtype)
            memory_enabled = self.memory_len > 0

            core = inputs
            with _record_function("GTrXLModule/input_proj"):
                if self.use_input_proj:
                    core = F.relu(self.input_proj(core))

            with _record_function("GTrXLModule/positional_encoding"):
                core = self.positional_encoding(core)
                core = self.dropout(core)

            layer_outputs: List[torch.Tensor] = []
            for layer_idx, layer in enumerate(self.layers):
                with _record_function(f"GTrXLModule/layer_{layer_idx}"):
                    mem = layer_mems[layer_idx]
                    mem_len = mem.size(0)
                    if mem_len > 0:
                        if mem.size(1) != batch_size:
                            mem = mem[:, :batch_size].contiguous()
                        combined = torch.cat([mem, core], dim=0)
                    else:
                        combined = core

                    attn_mask = None
                    if self.use_causal_mask:
                        total_len = combined.size(0)
                        attn_mask = self._get_causal_mask(total_len, device)

                    if self.use_activation_checkpoint and combined.requires_grad:

                        def _layer_run(inp, *, _layer=layer, _mask=attn_mask):
                            return _layer(inp, _mask)

                        layer_out = checkpoint(_layer_run, combined, use_reentrant=False)
                    else:
                        layer_out = layer(combined, attn_mask)
                    layer_outputs.append(layer_out)

                    if mem_len > 0:
                        core = layer_out[mem_len:]
                    else:
                        core = layer_out

            with _record_function("GTrXLModule/output_norm"):
                core = self.output_norm(core)
                core = self.dropout(core)

            if squeeze:
                core = core.squeeze(0)

            with _record_function("GTrXLModule/update_memory"):
                new_memory = update_memory_window(
                    layer_outputs,
                    layer_mems if memory_enabled else None,
                    self.memory_len,
                )
            return core, {"hidden_states": new_memory if memory_enabled else None}

    def initialize_memory(self, batch_size: int) -> Dict[str, Optional[List[torch.Tensor]]]:
        if self.memory_len <= 0:
            return {"hidden_states": None}
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        return {"hidden_states": empty_memory(self.n_layers, batch_size, self.d_model, device, dtype)}

    def _get_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        key = (size, device)
        mask = self._mask_cache.get(key)
        if mask is None:
            mask = torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)
            self._mask_cache[key] = mask
        return mask


@dataclass
class GTrXLConfig:
    """Backbone parameters for the GTrXL transformer."""

    hidden_size: int = 32
    latent_size: int | None = None
    num_layers: int = 1
    n_heads: int = 2
    d_ff: int = 128
    max_seq_len: int = 256
    memory_len: int = 32
    dropout: float = 0.05
    attn_dropout: float = 0.05
    positional_scale: float = 0.1
    use_gating: bool = True
    activation_checkpoint: bool = False
    use_flash_checkpoint: bool = False
    allow_tf32: bool = True
    use_fused_layernorm: bool = False

    def __post_init__(self) -> None:
        if self.latent_size is None:
            self.latent_size = self.hidden_size

    def build(self) -> GTrXLModule:
        """Construct the GTrXL backbone module."""

        return GTrXLModule(
            d_model=self.hidden_size,
            n_heads=self.n_heads,
            n_layers=self.num_layers,
            d_ff=self.d_ff,
            max_seq_len=self.max_seq_len,
            memory_len=self.memory_len,
            dropout=self.dropout,
            use_gating=self.use_gating,
            positional_scale=self.positional_scale,
            attn_dropout=self.attn_dropout,
            activation_checkpoint=self.activation_checkpoint,
            use_flash_checkpoint=self.use_flash_checkpoint,
            use_fused_layernorm=self.use_fused_layernorm,
            allow_tf32=self.allow_tf32,
        )

    def policy_defaults(self) -> dict[str, object]:
        """Return default policy-level overrides for this variant."""

        return {
            "manual_init": False,
            "strict_attr_indices": False,
            "learning_rate_hint": 7.5e-4,
        }


__all__ = ["GTrXLModule", "GTrXLConfig"]
