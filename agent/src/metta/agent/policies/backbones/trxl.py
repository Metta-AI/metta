from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from metta.agent.policies.backbones.transformer_utils import (
    _make_layer_norm,
    _record_function,
    empty_memory,
    normalize_memory,
    update_memory_window,
)


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
    ) -> None:
        super().__init__()
        self.pre_lnorm = pre_lnorm
        self.layer_norm = _make_layer_norm(d_model, False)
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
        self.layer_norm = _make_layer_norm(d_model, False)

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
    ) -> None:
        super().__init__(n_head, d_model, d_head, dropout, dropatt, pre_lnorm)
        self.r_net = nn.Linear(d_model, n_head * d_head, bias=False)

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
                content = self.layer_norm(content)
            r_head_k = self.r_net(rel_pos)
            w_head_q, w_head_k, w_head_v = torch.chunk(self.qkv_net(content), 3, dim=-1)

        klen = w_head_k.size(0)
        w_head_q = w_head_q.view(qlen, batch_size, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, batch_size, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, batch_size, self.n_head, self.d_head)

        if r_head_k.size(0) != rlen:
            r_head_k = r_head_k[-rlen:]
        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)

        rw_head_q = w_head_q + r_w_bias[None]
        AC = torch.einsum("ibnd,jbnd->bnij", (rw_head_q, w_head_k))

        rr_head_q = w_head_q + r_r_bias[None]
        BD = torch.einsum("ibnd,jnd->bnij", (rr_head_q, r_head_k))
        BD = self._rel_shift(BD)

        attn_score = (AC + BD) * self.scale

        if attn_mask is not None and attn_mask.any():
            if attn_mask.dim() == 2:
                attn_score = (
                    attn_score.float().masked_fill(attn_mask[None, None, :, :], float("-inf")).type_as(attn_score)
                )
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(attn_mask[:, None, :, :], float("-inf")).type_as(attn_score)

        attn_prob = F.softmax(attn_score, dim=-1)
        attn_prob = self.dropatt(attn_prob)

        attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, w_head_v)
        attn_vec = attn_vec.contiguous().view(qlen, batch_size, self.n_head * self.d_head)

        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            return content + attn_out
        return self.layer_norm(content + attn_out)


class XLRelPartialLearnableDecoderLayer(nn.Module):
    """Single Transformer-XL decoder layer."""

    def __init__(
        self,
        n_head: int,
        d_model: int,
        d_head: int,
        d_ff: int,
        dropout: float,
        dropatt: float,
        pre_lnorm: bool,
    ) -> None:
        super().__init__()
        self.attn = XLRelPartialLearnableMultiHeadAttn(n_head, d_model, d_head, dropout, dropatt, pre_lnorm)
        self.ff = XLPositionwiseFF(d_model, d_ff, dropout, pre_lnorm)

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
        return self.ff(output)


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
        *,
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
        if use_flash_checkpoint:
            warnings.warn(
                "TransformerXLModule ignores use_flash_checkpoint; set this to False to silence the warning.",
                stacklevel=2,
            )
        if use_fused_layernorm:
            warnings.warn(
                "TransformerXLModule ignores use_fused_layernorm; set this to False to silence the warning.",
                stacklevel=2,
            )

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
        with _record_function("TransformerXLModule/forward"):
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
                            mem_layer if mem_layer is not None else core_out.new_zeros(0, batch_size, self.d_model)
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


@dataclass
class TRXLConfig:
    """Backbone parameters for the Transformer-XL variant."""

    hidden_size: int = 32
    latent_size: int | None = None
    num_layers: int = 1
    n_heads: int = 2
    d_ff: int = 128
    max_seq_len: int = 192
    memory_len: int = 32
    dropout: float = 0.05
    attn_dropout: float = 0.05
    pre_lnorm: bool = True
    same_length: bool = False
    clamp_len: int = -1
    ext_len: int = 0
    activation_checkpoint: bool = False
    use_flash_checkpoint: bool = False
    allow_tf32: bool = True
    use_fused_layernorm: bool = False

    def __post_init__(self) -> None:
        if self.latent_size is None:
            self.latent_size = self.hidden_size

    def build(self) -> TransformerXLModule:
        """Construct the Transformer-XL backbone module."""

        return TransformerXLModule(
            d_model=self.hidden_size,
            n_heads=self.n_heads,
            n_layers=self.num_layers,
            d_ff=self.d_ff,
            max_seq_len=self.max_seq_len,
            memory_len=self.memory_len,
            dropout=self.dropout,
            dropatt=self.attn_dropout,
            pre_lnorm=self.pre_lnorm,
            same_length=self.same_length,
            clamp_len=self.clamp_len,
            ext_len=self.ext_len,
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
            "learning_rate_hint": 9.0e-4,
        }


__all__ = ["TransformerXLModule", "TRXLConfig"]
