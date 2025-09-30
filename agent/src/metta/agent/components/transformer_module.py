"""Transformer modules for Metta transformer policies."""

from __future__ import annotations

import contextlib
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:  # pragma: no cover - optional dependency
    from apex.normalization.fused_layer_norm import FusedLayerNorm  # type: ignore
except ImportError:  # pragma: no cover
    FusedLayerNorm = None

try:  # pragma: no cover - optional dependency
    from flash_attn.flash_attn_interface import flash_attn_func  # type: ignore
except ImportError:  # pragma: no cover
    flash_attn_func = None


def _record_function(name: str):
    profiler_mod = getattr(torch, "profiler", None)
    if profiler_mod is not None and hasattr(profiler_mod, "record_function"):
        return profiler_mod.record_function(name)
    return contextlib.nullcontext()


def _make_layer_norm(d_model: int, use_fused: bool) -> nn.Module:
    if use_fused and FusedLayerNorm is not None:
        return FusedLayerNorm(d_model)
    return nn.LayerNorm(d_model)


class TF32Context:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled and torch.cuda.is_available()
        self.prev = None

    def __enter__(self):
        if self.enabled:
            self.prev = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled and self.prev is not None:
            torch.backends.cuda.matmul.allow_tf32 = self.prev


def empty_memory(
    num_layers: int,
    batch_size: int,
    d_model: int,
    device: torch.device,
    dtype: torch.dtype,
) -> List[torch.Tensor]:
    """Return a list of empty memory tensors."""

    with _record_function("Transformer/empty_memory"):
        return [torch.zeros(0, batch_size, d_model, device=device, dtype=dtype) for _ in range(num_layers)]


def normalize_memory(
    memory_len: int,
    num_layers: int,
    memory: Optional[Sequence[torch.Tensor]],
    batch_size: int,
    d_model: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[List[torch.Tensor]]:
    """Normalize previously stored memory tensors to the expected shape."""
    with _record_function("Transformer/normalize_memory"):
        if memory_len <= 0:
            return None

        if memory is None or len(memory) != num_layers:
            return empty_memory(num_layers, batch_size, d_model, device, dtype)

        normalized: List[torch.Tensor] = []
        for tensor in memory:
            if tensor is None or tensor.numel() == 0:
                normalized.append(torch.zeros(0, batch_size, d_model, device=device, dtype=dtype))
                continue
            mem = tensor.to(device=device, dtype=dtype)
            if mem.size(1) != batch_size:
                mem = mem[:, :batch_size].contiguous()
            normalized.append(mem)
        return normalized


def update_memory_window(
    layer_outputs: Sequence[torch.Tensor],
    previous_memory: Optional[Sequence[torch.Tensor]],
    memory_len: int,
    ext_len: int = 0,
) -> Optional[List[torch.Tensor]]:
    """Return the updated memory window for each layer."""
    with _record_function("Transformer/update_memory_window"):
        if memory_len <= 0:
            return None

        if not layer_outputs:
            return [torch.zeros(0)] * 0

        device = layer_outputs[0].device
        dtype = layer_outputs[0].dtype
        batch_size = layer_outputs[0].size(1)
        d_model = layer_outputs[0].size(2)
        num_layers = len(layer_outputs)

        if previous_memory is None or len(previous_memory) != num_layers:
            previous_memory = empty_memory(num_layers, batch_size, d_model, device, dtype)

        with torch.no_grad():
            mlen = previous_memory[0].size(0) if previous_memory else 0
            qlen = layer_outputs[0].size(0)
            end_idx = mlen + max(0, qlen - ext_len)
            beg_idx = max(0, end_idx - memory_len)

            updated: List[torch.Tensor] = []
            for prev, output in zip(previous_memory, layer_outputs, strict=False):
                cat = torch.cat([prev, output], dim=0)
                updated.append(cat[beg_idx:end_idx].detach())
        return updated


# ---------------------------------------------------------------------------
# Full-context GTrXL-style transformer (legacy working version)
# ---------------------------------------------------------------------------


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
        use_flash_checkpoint: bool = False,
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
        self._flash_available = flash_attn_func is not None
        self._attn_dropout_p = float(attn_dropout)
        self._dropout_p = float(dropout)
        self.use_flash_checkpoint = use_flash_checkpoint

        nn.init.xavier_uniform_(self.qkv_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=1.0)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len, batch_size, _ = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.view(seq_len, batch_size, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 1, 3, 0, 4)  # (3, batch, heads, seq, d_k)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if attn_mask is None and self.use_causal_mask and flash_attn_func is not None and x.is_cuda:
            dropout_p = self._attn_dropout_p if self.training else 0.0
            q_flash = q.permute(0, 3, 1, 2)  # (batch, seq, heads, d)
            k_flash = k.permute(0, 3, 1, 2)
            v_flash = v.permute(0, 3, 1, 2)
            def _flash(q_t, k_t, v_t):
                return flash_attn_func(
                    q_t,
                    k_t,
                    v_t,
                    dropout_p=dropout_p,
                    softmax_scale=None,
                    causal=True,
                )

            if self.use_flash_checkpoint and q_flash.requires_grad:
                out = checkpoint(_flash, q_flash, k_flash, v_flash, use_reentrant=False)
            else:
                out = _flash(q_flash, k_flash, v_flash)
            out = out.permute(1, 0, 2, 3).reshape(seq_len, batch_size, self.d_model)
            return self.out_proj(out)

        if attn_mask is None and hasattr(F, "scaled_dot_product_attention"):
            dropout_p = self._attn_dropout_p if self.training else 0.0
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=self.use_causal_mask,
            )
            out = out.permute(2, 0, 1, 3).reshape(seq_len, batch_size, self.d_model)
            return self.out_proj(out)

        # Fallback path with explicit masking or CPU execution
        q_2d = q.reshape(batch_size * self.n_heads, seq_len, self.d_k)
        k_2d = k.reshape(batch_size * self.n_heads, seq_len, self.d_k)
        v_2d = v.reshape(batch_size * self.n_heads, seq_len, self.d_k)

        scores = torch.bmm(q_2d, k_2d.transpose(1, 2)) / math.sqrt(self.d_k)

        if self.use_causal_mask:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                expanded = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                expanded = attn_mask
                if expanded.size(0) == 1:
                    expanded = expanded.expand(batch_size, -1, -1)
            else:
                raise ValueError("Attention mask must have dim 2 or 3.")
            expanded = expanded.to(device=x.device)
            expanded = expanded.unsqueeze(1).expand(batch_size, self.n_heads, seq_len, seq_len)
            scores = scores.view(batch_size, self.n_heads, seq_len, seq_len)
            scores = scores.masked_fill(expanded.to(torch.bool), float("-inf"))
            scores = scores.view(batch_size * self.n_heads, seq_len, seq_len)

        weights = F.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=self._attn_dropout_p if self.training else 0.0, training=self.training)

        out = torch.bmm(weights, v_2d)
        out = out.view(batch_size, self.n_heads, seq_len, self.d_k)
        out = out.permute(2, 0, 1, 3).reshape(seq_len, batch_size, self.d_model)
        return self.out_proj(out)


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
        use_flash_checkpoint: bool = False,
        use_fused_layernorm: bool = False,
    ) -> None:
        super().__init__()
        self.use_gating = use_gating
        self.attention = GTrXLMultiHeadSelfAttention(
            d_model,
            n_heads,
            dropout,
            use_causal_mask,
            attn_dropout=attn_dropout,
            use_flash_checkpoint=use_flash_checkpoint,
        )
        self.norm1 = _make_layer_norm(d_model, use_fused_layernorm)
        self.norm2 = _make_layer_norm(d_model, use_fused_layernorm)

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
        self.use_flash_checkpoint = use_flash_checkpoint
        self.use_fused_layernorm = use_fused_layernorm
        self.attn_dropout = attn_dropout
        self.allow_tf32 = allow_tf32

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
                    use_flash_checkpoint=self.use_flash_checkpoint,
                    use_fused_layernorm=self.use_fused_layernorm,
                )
                for _ in range(n_layers)
            ]
        )
        self.output_norm = _make_layer_norm(d_model, self.use_fused_layernorm)
        self.dropout = nn.Dropout(dropout)
        self._mask_cache: dict[tuple[int, torch.device], torch.Tensor] = {}

    def forward(
        self,
        inputs: torch.Tensor,
        memory: Optional[Dict[str, Optional[List[torch.Tensor]]]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Optional[List[torch.Tensor]]]]:
        with _record_function("GTrXLModule/forward"), TF32Context(self.allow_tf32):
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


# ---------------------------------------------------------------------------
# Transformer-XL implementation (improved variant)
# ---------------------------------------------------------------------------


class XLPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding identical to Transformer-XL."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0.0, d_model, 2.0) / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, positions: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
        sinusoid_inp = torch.outer(positions, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        if batch_size is not None:
            return pos_emb[:, None, :].expand(-1, batch_size, -1)
        return pos_emb[:, None, :]


class XLPositionwiseFF(nn.Module):
    """Position-wise feed-forward layer for Transformer-XL."""

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        dropout: float,
        pre_lnorm: bool,
        *,
        use_fused_layernorm: bool = False,
    ) -> None:
        super().__init__()
        self.pre_lnorm = pre_lnorm
        self.layer_norm = _make_layer_norm(d_model, use_fused_layernorm)
        self.core = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.pre_lnorm:
            output = inputs + self.core(self.layer_norm(inputs))
        else:
            output = self.layer_norm(inputs + self.core(inputs))
        return output


class XLRelMultiHeadAttn(nn.Module):
    """Base class for relative multi-head attention."""

    def __init__(
        self,
        n_head: int,
        d_model: int,
        d_head: int,
        dropout: float,
        dropatt: float,
        pre_lnorm: bool,
        *,
        use_fused_layernorm: bool = False,
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.scale = 1.0 / math.sqrt(d_head)
        self.pre_lnorm = pre_lnorm

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.layer_norm = _make_layer_norm(d_model, use_fused_layernorm)

    @staticmethod
    def _rel_shift(x: torch.Tensor) -> torch.Tensor:
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)
        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])
        return x_padded[1:].view_as(x)

    def forward(
        self,
        content: torch.Tensor,
        rel_pos: torch.Tensor,
        r_w_bias: torch.Tensor,
        r_r_bias: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        mems: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class XLRelPartialLearnableMultiHeadAttn(XLRelMultiHeadAttn):
    """Partial relative position multi-head attention used by Transformer-XL."""

    def __init__(
        self,
        n_head: int,
        d_model: int,
        d_head: int,
        dropout: float,
        dropatt: float,
        pre_lnorm: bool,
        *,
        use_flash_checkpoint: bool = False,
        use_fused_layernorm: bool = False,
    ) -> None:
        super().__init__(
            n_head,
            d_model,
            d_head,
            dropout,
            dropatt,
            pre_lnorm,
            use_fused_layernorm=use_fused_layernorm,
        )
        self.r_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.layer_norm = _make_layer_norm(d_model, use_fused_layernorm)
        self.use_flash_checkpoint = use_flash_checkpoint

    def forward(
        self,
        content: torch.Tensor,
        rel_pos: torch.Tensor,
        r_w_bias: torch.Tensor,
        r_r_bias: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        mems: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qlen, rlen, batch_size = content.size(0), rel_pos.size(0), content.size(1)

        if mems is not None:
            cat = torch.cat([mems, content], dim=0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(rel_pos)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(content))
            else:
                w_heads = self.qkv_net(content)
            r_head_k = self.r_net(rel_pos)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)
        w_head_q = w_head_q.view(qlen, batch_size, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, batch_size, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, batch_size, self.n_head, self.d_head)
        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)

        if hasattr(F, "scaled_dot_product_attention") and w_head_q.is_cuda:
            try:
                q_tilde = w_head_q + r_w_bias[None, None, :, :]
                q_sdpa = q_tilde.permute(1, 2, 0, 3).contiguous()
                k_sdpa = w_head_k.permute(1, 2, 0, 3).contiguous()
                v_sdpa = w_head_v.permute(1, 2, 0, 3).contiguous()

                rr_head_q = w_head_q + r_r_bias[None, None, :, :]
                BD = torch.einsum("ibnd,jnd->ijbn", rr_head_q, r_head_k)
                BD = self._rel_shift(BD)
                attn_bias = BD.permute(2, 3, 0, 1).contiguous() * self.scale

                if attn_mask is not None:
                    if attn_mask.dim() == 2:
                        mask = attn_mask[None, None, :, :]
                    elif attn_mask.dim() == 3:
                        mask = attn_mask[:, None, :, :]
                    else:
                        raise ValueError("Attention mask must have dim 2 or 3.")
                    mask = mask.to(attn_bias.device)
                    attn_bias = attn_bias.masked_fill(mask.bool(), float("-inf"))

                dropout_p = self.dropatt.p if self.training else 0.0
                attn_out = F.scaled_dot_product_attention(
                    q_sdpa,
                    k_sdpa,
                    v_sdpa,
                    attn_mask=attn_bias,
                    dropout_p=dropout_p,
                    is_causal=False,
                )
                attn_out = attn_out.permute(2, 0, 1, 3).reshape(qlen, batch_size, self.n_head * self.d_head)
                attn_out = self.drop(self.o_net(attn_out))

                if self.pre_lnorm:
                    return content + attn_out
                return self.layer_norm(content + attn_out)
            except RuntimeError:
                pass

        rw_head_q = w_head_q + r_w_bias
        AC = torch.einsum("ibnd,jbnd->ijbn", (rw_head_q, w_head_k))

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum("ibnd,jnd->ijbn", (rr_head_q, r_head_k))
        BD = self._rel_shift(BD)

        attn_score = (AC + BD) * self.scale

        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = (
                    attn_score.float().masked_fill(attn_mask[None, :, :, None], float("-inf")).type_as(attn_score)
                )
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(attn_mask[:, :, :, None], float("-inf")).type_as(attn_score)

        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        attn_vec = torch.einsum("ijbn,jbnd->ibnd", (attn_prob, w_head_v))
        attn_vec = attn_vec.contiguous().view(qlen, batch_size, self.n_head * self.d_head)
        attn_out = self.drop(self.o_net(attn_vec))

        if self.pre_lnorm:
            return content + attn_out
        return self.layer_norm(content + attn_out)


class XLRelPartialLearnableDecoderLayer(nn.Module):
    """Transformer-XL decoder layer."""

    def __init__(
        self,
        n_head: int,
        d_model: int,
        d_head: int,
        d_inner: int,
        dropout: float,
        dropatt: float,
        pre_lnorm: bool,
        *,
        use_flash_checkpoint: bool = False,
        use_fused_layernorm: bool = False,
    ) -> None:
        super().__init__()
        self.attn = XLRelPartialLearnableMultiHeadAttn(
            n_head,
            d_model,
            d_head,
            dropout,
            dropatt,
            pre_lnorm,
            use_flash_checkpoint=use_flash_checkpoint,
            use_fused_layernorm=use_fused_layernorm,
        )
        self.ff = XLPositionwiseFF(
            d_model,
            d_inner,
            dropout,
            pre_lnorm,
            use_fused_layernorm=use_fused_layernorm,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        rel_pos: torch.Tensor,
        r_w_bias: torch.Tensor,
        r_r_bias: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        mems: Optional[torch.Tensor],
    ) -> torch.Tensor:
        output = self.attn(inputs, rel_pos, r_w_bias, r_r_bias, attn_mask, mems)
        output = self.ff(output)
        return output


class TransformerXLModule(nn.Module):
    """Transformer-XL style module that exposes the legacy interface."""

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 256,
        memory_len: int = 64,
        dropout: float = 0.1,
        dropatt: float = 0.1,
        pre_lnorm: bool = True,
        same_length: bool = False,
        clamp_len: int = -1,
        ext_len: int = 0,
        attn_type: int = 0,
        activation_checkpoint: bool = False,
        use_flash_checkpoint: bool = False,
        use_fused_layernorm: bool = False,
        allow_tf32: bool = True,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if attn_type != 0:
            raise NotImplementedError("Only relative positional attention (attn_type=0) is supported.")

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.memory_len = memory_len
        self.pre_lnorm = pre_lnorm
        self.same_length = same_length
        self.clamp_len = clamp_len
        self.ext_len = ext_len
        self.attn_type = attn_type
        self.use_activation_checkpoint = activation_checkpoint
        self.use_flash_checkpoint = use_flash_checkpoint
        self.use_fused_layernorm = use_fused_layernorm
        self.allow_tf32 = allow_tf32
        self.use_activation_checkpoint = activation_checkpoint
        self.use_flash_checkpoint = use_flash_checkpoint
        self.use_fused_layernorm = use_fused_layernorm
        self.allow_tf32 = allow_tf32

        d_head = d_model // n_heads

        self.pos_emb = XLPositionalEmbedding(d_model)
        self.r_w_bias = nn.Parameter(torch.zeros(n_heads, d_head))
        self.r_r_bias = nn.Parameter(torch.zeros(n_heads, d_head))
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                XLRelPartialLearnableDecoderLayer(
                    n_heads,
                    d_model,
                    d_head,
                    d_ff,
                    dropout,
                    dropatt,
                    pre_lnorm,
                    use_flash_checkpoint=self.use_flash_checkpoint,
                    use_fused_layernorm=self.use_fused_layernorm,
                )
                for _ in range(n_layers)
            ]
        )
        nn.init.normal_(self.r_w_bias, mean=0.0, std=0.02)
        nn.init.normal_(self.r_r_bias, mean=0.0, std=0.02)
        self._attn_mask_cache: dict[tuple[int, int, bool, torch.device], torch.Tensor] = {}
        self._pos_seq_cache: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}

    def forward(
        self, inputs: torch.Tensor, memory: Optional[Dict[str, List[torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Optional[List[torch.Tensor]]]]:
        with _record_function("TransformerXLModule/forward"), TF32Context(self.allow_tf32):
            if inputs.dim() == 2:
                inputs = inputs.unsqueeze(0)
            if inputs.dim() != 3:
                raise ValueError(f"Expected tensor of shape (T, B, D), received {inputs.shape}")

            seq_len, batch_size, _ = inputs.shape
            device = inputs.device
            dtype = inputs.dtype

            with _record_function("TransformerXLModule/normalize_memory"):
                stored_memory = memory.get("hidden_states") if isinstance(memory, dict) else None
                mems = normalize_memory(
                    self.memory_len,
                    self.n_layers,
                    stored_memory,
                    batch_size,
                    self.d_model,
                    device,
                    dtype,
                )
            mem_enabled = mems is not None
            if mems is None:
                mems = empty_memory(self.n_layers, batch_size, self.d_model, device, dtype)
            mlen = mems[0].size(0) if mems else 0
            klen = mlen + seq_len

            with _record_function("TransformerXLModule/attn_mask"):
                attn_mask = self._get_attn_mask(seq_len, mlen, klen, device)
                attn_mask = attn_mask[:, :, None]

            with _record_function("TransformerXLModule/positional"):
                pos_seq = self._get_pos_seq(klen, device, dtype)
                if self.clamp_len > 0:
                    pos_seq = pos_seq.clamp(max=float(self.clamp_len))
                pos_emb = self.pos_emb(pos_seq)
                pos_emb = self.drop(pos_emb)

            core_out = self.drop(inputs)
            hiddens: List[torch.Tensor] = [core_out]

            for layer_id, layer in enumerate(self.layers):
                with _record_function(f"TransformerXLModule/layer_{layer_id}"):
                    mem_layer = mems[layer_id] if mem_enabled else None

                    if self.use_activation_checkpoint and core_out.requires_grad:
                        placeholder = (
                            mem_layer
                            if mem_layer is not None
                            else core_out.new_zeros(0, batch_size, self.d_model)
                        )

                        def _layer_run(
                            inp,
                            mem,
                            _layer=layer,
                            _pos=pos_emb,
                            _rw=self.r_w_bias,
                            _rr=self.r_r_bias,
                            _mask=attn_mask,
                        ):
                            mem_in = None if mem.size(0) == 0 else mem
                            return _layer(inp, _pos, _rw, _rr, _mask, mem_in)

                        core_out = checkpoint(_layer_run, core_out, placeholder, use_reentrant=False)
                    else:
                        core_out = layer(core_out, pos_emb, self.r_w_bias, self.r_r_bias, attn_mask, mem_layer)
                    hiddens.append(core_out)

            with _record_function("TransformerXLModule/post"):
                core_out = self.drop(core_out)
                new_memory = update_memory_window(
                    hiddens[1:],
                    mems if mem_enabled else None,
                    self.memory_len,
                    ext_len=self.ext_len,
                )
            return core_out, {"hidden_states": new_memory if mem_enabled else None}

    def initialize_memory(self, batch_size: int) -> Dict[str, Optional[List[torch.Tensor]]]:
        if self.memory_len <= 0:
            return {"hidden_states": None}
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        return {"hidden_states": empty_memory(self.n_layers, batch_size, self.d_model, device, dtype)}

    def _get_attn_mask(self, seq_len: int, mem_len: int, klen: int, device: torch.device) -> torch.Tensor:
        key = (seq_len, mem_len, self.same_length, device)
        mask = self._attn_mask_cache.get(key)
        if mask is None:
            if self.same_length:
                all_ones = torch.ones(seq_len, klen, device=device, dtype=torch.bool)
                mask_len = klen - self.memory_len
                mask_shift_len = seq_len - mask_len if mask_len > 0 else seq_len
                mask = torch.triu(all_ones, diagonal=1 + mem_len) | torch.tril(all_ones, diagonal=-mask_shift_len)
            else:
                mask = torch.triu(torch.ones(seq_len, klen, device=device, dtype=torch.bool), diagonal=1 + mem_len)
            self._attn_mask_cache[key] = mask
        return mask

    def _get_pos_seq(self, klen: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (klen, device, dtype)
        pos_seq = self._pos_seq_cache.get(key)
        if pos_seq is None:
            pos_seq = torch.arange(klen - 1, -1, -1.0, device=device, dtype=dtype)
            self._pos_seq_cache[key] = pos_seq
        return pos_seq
