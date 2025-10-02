from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from metta.agent.components.transformer_utils import empty_memory, normalize_memory, update_memory_window


class NvidiaPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding identical to NVIDIA's Transformer-XL."""

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


class NvidiaPositionwiseFF(nn.Module):
    """Feed-forward layer used in NVIDIA's Transformer-XL implementation."""

    def __init__(self, d_model: int, d_inner: int, dropout: float, pre_lnorm: bool) -> None:
        super().__init__()
        self.pre_lnorm = pre_lnorm
        self.layer_norm = nn.LayerNorm(d_model)
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


class NvidiaRelPartialLearnableMultiHeadAttn(nn.Module):
    """Relative multi-head attention with learnable biases (NVIDIA variant)."""

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
        self.d_head = d_head
        self.pre_lnorm = pre_lnorm
        self.scale = 1.0 / math.sqrt(d_head)

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)
        self.r_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.layer_norm = nn.LayerNorm(d_model)

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
        qlen, batch_size = content.size(0), content.size(1)

        if mems is not None and mems.numel() > 0:
            cat = torch.cat([mems, content], dim=0)
            if self.pre_lnorm:
                cat = self.layer_norm(cat)
            w_heads = self.qkv_net(cat)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                content = self.layer_norm(content)
            w_head_q, w_head_k, w_head_v = torch.chunk(self.qkv_net(content), 3, dim=-1)

        klen = w_head_k.size(0)
        w_head_q = w_head_q.view(qlen, batch_size, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, batch_size, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, batch_size, self.n_head, self.d_head)

        r_head_k = self.r_net(rel_pos).view(rel_pos.size(0), self.n_head, self.d_head)
        if klen > r_head_k.size(0):
            pad = r_head_k[0:1].expand(klen - r_head_k.size(0), -1, -1)
            r_head_k = torch.cat([pad, r_head_k], dim=0)
        else:
            r_head_k = r_head_k[-klen:]

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


class NvidiaRelPartialLearnableDecoderLayer(nn.Module):
    def __init__(
        self,
        n_head: int,
        d_model: int,
        d_head: int,
        d_inner: int,
        dropout: float,
        dropatt: float,
        pre_lnorm: bool,
    ) -> None:
        super().__init__()
        self.attn = NvidiaRelPartialLearnableMultiHeadAttn(n_head, d_model, d_head, dropout, dropatt, pre_lnorm)
        self.ff = NvidiaPositionwiseFF(d_model, d_inner, dropout, pre_lnorm)

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


class NvidiaTransformerCore(nn.Module):
    """Minimal Transformer-XL core copied from NVIDIA's reference implementation."""

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_inner: int,
        mem_len: int,
        dropout: float,
        dropatt: float,
        pre_lnorm: bool,
        clamp_len: int,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.mem_len = mem_len
        self.clamp_len = clamp_len
        self.pre_lnorm = pre_lnorm

        d_head = d_model // n_heads

        self.pos_emb = NvidiaPositionalEmbedding(d_model)
        self.r_w_bias = nn.Parameter(torch.zeros(n_heads, d_head))
        self.r_r_bias = nn.Parameter(torch.zeros(n_heads, d_head))
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                NvidiaRelPartialLearnableDecoderLayer(
                    n_heads,
                    d_model,
                    d_head,
                    d_inner,
                    dropout,
                    dropatt,
                    pre_lnorm,
                )
                for _ in range(n_layers)
            ]
        )
        nn.init.normal_(self.r_w_bias, mean=0.0, std=0.02)
        nn.init.normal_(self.r_r_bias, mean=0.0, std=0.02)

    def forward(
        self,
        inputs: torch.Tensor,
        memory: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0)
        if inputs.dim() != 3:
            raise ValueError(f"Expected tensor of shape (T, B, D), received {inputs.shape}")

        seq_len, batch_size, _ = inputs.shape
        device = inputs.device
        dtype = inputs.dtype

        normalized = memory
        if normalized is None or len(normalized) != self.n_layers:
            normalized = empty_memory(self.n_layers, batch_size, self.d_model, device, dtype)

        mem_enabled = normalized is not None
        mem_list = normalized if mem_enabled else None
        mlen = normalized[0].size(0) if normalized else 0
        klen = mlen + seq_len

        attn_mask = torch.triu(torch.ones(seq_len, klen, device=device, dtype=torch.bool), diagonal=1 + mlen)
        attn_mask = attn_mask[:, :, None]

        pos_seq = torch.arange(klen - 1, -1, -1.0, device=device, dtype=dtype)
        if self.clamp_len > 0:
            pos_seq = pos_seq.clamp(max=float(self.clamp_len))
        pos_emb = self.pos_emb(pos_seq)
        pos_emb = self.drop(pos_emb)

        core_out = self.drop(inputs)
        hiddens: List[torch.Tensor] = [core_out]

        for layer, mem_layer in zip(self.layers, mem_list or [], strict=False):
            mem_in = None if mem_layer.size(0) == 0 else mem_layer
            core_out = layer(core_out, pos_emb, self.r_w_bias, self.r_r_bias, attn_mask, mem_in)
            hiddens.append(core_out)

        core_out = self.drop(core_out)
        new_memory = update_memory_window(hiddens[1:], mem_list, self.mem_len)
        return core_out, new_memory

    def initialize_memory(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> List[torch.Tensor]:
        return empty_memory(self.n_layers, batch_size, self.d_model, device, dtype)


class NvidiaTransformerModule(nn.Module):
    """Wrapper exposing the same interface as TransformerModule but using NVIDIA blocks."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int,
        memory_len: int,
        dropout: float = 0.1,
        dropatt: float = 0.0,
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
        if same_length:
            raise NotImplementedError("same_length masking is not implemented for the NVIDIA transformer module")
        if attn_type != 0:
            raise NotImplementedError("Only relative positional attention (attn_type=0) is supported.")

        self.use_flash_checkpoint = use_flash_checkpoint
        self.use_fused_layernorm = use_fused_layernorm
        self.allow_tf32 = allow_tf32
        if use_flash_checkpoint:
            warnings.warn(
                "NvidiaTransformerModule ignores use_flash_checkpoint; set this to False to silence the warning.",
                stacklevel=2,
            )
        if use_fused_layernorm:
            warnings.warn(
                "NvidiaTransformerModule ignores use_fused_layernorm; set this to False to silence the warning.",
                stacklevel=2,
            )

        self.core = NvidiaTransformerCore(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_inner=d_ff,
            mem_len=memory_len,
            dropout=dropout,
            dropatt=dropatt,
            pre_lnorm=pre_lnorm,
            clamp_len=clamp_len,
        )
        self.output_dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.n_layers = n_layers
        self.memory_len = memory_len

    def forward(
        self,
        inputs: torch.Tensor,
        memory: Optional[Dict[str, Optional[List[torch.Tensor]]]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Optional[List[torch.Tensor]]]]:
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0)
        if inputs.dim() != 3:
            raise ValueError(f"Expected tensor of shape (T, B, D), received {inputs.shape}")

        stored_memory = memory.get("hidden_states") if isinstance(memory, dict) else None
        normalized = normalize_memory(
            self.memory_len,
            self.n_layers,
            stored_memory,
            inputs.size(1),
            self.d_model,
            inputs.device,
            inputs.dtype,
        )
        mem_enabled = normalized is not None
        mem_list = normalized if mem_enabled else None

        core_out, new_memory = self.core(inputs, mem_list)
        core_out = self.output_dropout(core_out)
        return core_out, {"hidden_states": new_memory if mem_enabled and new_memory else None}

    def initialize_memory(self, batch_size: int) -> Dict[str, Optional[List[torch.Tensor]]]:
        if self.memory_len <= 0:
            return {"hidden_states": None}
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        hidden_states = self.core.initialize_memory(batch_size, device=device, dtype=dtype)
        return {"hidden_states": hidden_states if hidden_states else None}


@dataclass
class TRXLNvidiaConfig:
    """Backbone parameters for the NVIDIA-optimized Transformer-XL variant."""

    hidden_size: int = 48
    latent_size: int | None = None
    num_layers: int = 2
    n_heads: int = 2
    d_ff: int = 192
    max_seq_len: int = 192
    memory_len: int = 32
    dropout: float = 0.05
    attn_dropout: float = 0.0
    pre_lnorm: bool = False
    clamp_len: int = -1
    ext_len: int = 0
    activation_checkpoint: bool = False
    use_flash_checkpoint: bool = False
    allow_tf32: bool = True
    use_fused_layernorm: bool = False

    def __post_init__(self) -> None:
        if self.latent_size is None:
            self.latent_size = self.hidden_size

    variant: str = "trxl_nvidia"

    def build(self) -> NvidiaTransformerModule:
        """Construct the NVIDIA Transformer-XL backbone module."""

        return NvidiaTransformerModule(
            d_model=self.hidden_size,
            n_heads=self.n_heads,
            n_layers=self.num_layers,
            d_ff=self.d_ff,
            max_seq_len=self.max_seq_len,
            memory_len=self.memory_len,
            dropout=self.dropout,
            dropatt=self.attn_dropout,
            pre_lnorm=self.pre_lnorm,
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
            "manual_init": True,
            "strict_attr_indices": True,
            "learning_rate_hint": 3.0e-4,
        }


__all__ = ["NvidiaTransformerModule", "TRXLNvidiaConfig"]
