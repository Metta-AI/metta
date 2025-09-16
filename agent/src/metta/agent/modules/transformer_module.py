"""Transformer-XL based sequence module used by PyTorch transformer agents."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding identical to Transformer-XL."""

    def __init__(self, d_model: int):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0.0, d_model, 2.0) / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, positions: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
        sinusoid_inp = torch.outer(positions, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        if batch_size is not None:
            return pos_emb[:, None, :].expand(-1, batch_size, -1)
        return pos_emb[:, None, :]


class PositionwiseFF(nn.Module):
    """Position-wise feed-forward network with optional pre-layernorm."""

    def __init__(self, d_model: int, d_inner: int, dropout: float, pre_lnorm: bool) -> None:
        super().__init__()
        self.core = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.pre_lnorm = pre_lnorm

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.pre_lnorm:
            output = self.core(self.layer_norm(inputs)) + inputs
        else:
            output = self.layer_norm(inputs + self.core(inputs))
        return output


class RelMultiHeadAttn(nn.Module):
    """Relative position multi-head attention from Transformer-XL."""

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
        self.layer_norm = nn.LayerNorm(d_model)

    def _rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)
        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])
        x = x_padded[1:].view_as(x)
        return x

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


class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    """Partial relative attention variant used by Transformer-XL."""

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

        if mems is not None and mems.numel() > 0:
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

        rw_head_q = w_head_q + r_w_bias
        AC = torch.einsum("ibnd,jbnd->ijbn", rw_head_q, w_head_k)

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum("ibnd,jnd->ijbn", rr_head_q, r_head_k)
        BD = self._rel_shift(BD)

        attn_score = (AC + BD) * self.scale

        if attn_mask is not None and attn_mask.any():
            if attn_mask.dim() == 2:
                attn_score = (
                    attn_score.float().masked_fill(attn_mask[None, :, :, None], float("-inf")).type_as(attn_score)
                )
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(attn_mask[:, :, :, None], float("-inf")).type_as(attn_score)

        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        attn_vec = torch.einsum("ijbn,jbnd->ibnd", attn_prob, w_head_v)
        attn_vec = attn_vec.contiguous().view(qlen, batch_size, self.n_head * self.d_head)

        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            return content + attn_out
        return self.layer_norm(content + attn_out)


class RelPartialLearnableDecoderLayer(nn.Module):
    """Decoder layer that applies relative multi-head attention and a feed-forward block."""

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
        self.attn = RelPartialLearnableMultiHeadAttn(n_head, d_model, d_head, dropout, dropatt, pre_lnorm)
        self.ff = PositionwiseFF(d_model, d_inner, dropout, pre_lnorm)

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


class TransformerModule(nn.Module):
    """Transformer-XL style module that exposes the interface used by Metta agents."""

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
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.memory_len = memory_len
        self.pre_lnorm = pre_lnorm
        self.clamp_len = max_seq_len

        d_head = d_model // n_heads

        self.pos_emb = PositionalEmbedding(d_model)
        self.r_w_bias = nn.Parameter(torch.zeros(n_heads, d_head))
        self.r_r_bias = nn.Parameter(torch.zeros(n_heads, d_head))
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                RelPartialLearnableDecoderLayer(
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
        self.final_norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.r_w_bias, mean=0.0, std=0.02)
        nn.init.normal_(self.r_r_bias, mean=0.0, std=0.02)

    def forward(
        self, inputs: torch.Tensor, memory: Optional[Dict[str, List[torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Optional[List[torch.Tensor]]]]:
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0)
        if inputs.dim() != 3:
            raise ValueError(f"Expected tensor of shape (T, B, D), received {inputs.shape}")

        seq_len, batch_size, _ = inputs.shape
        device = inputs.device
        dtype = inputs.dtype

        mems = None
        if memory is not None and memory.get("hidden_states"):
            candidate = [m.to(device) for m in memory["hidden_states"] if isinstance(m, torch.Tensor)]
            if len(candidate) == self.n_layers:
                mems = candidate
                if mems[0].size(1) != batch_size:
                    mems = [m[:, :batch_size].contiguous() for m in mems]
        else:
            mems = [torch.zeros(0, batch_size, self.d_model, device=device, dtype=dtype) for _ in range(self.n_layers)]
        if not mems or len(mems) != self.n_layers:
            mems = [torch.zeros(0, batch_size, self.d_model, device=device, dtype=dtype) for _ in range(self.n_layers)]

        mlen = mems[0].size(0) if mems else 0
        klen = mlen + seq_len
        pos_seq = torch.arange(klen - 1, -1, -1.0, device=device, dtype=dtype)
        if self.clamp_len > 0:
            pos_seq = pos_seq.clamp(max=float(self.clamp_len))
        pos_emb = self.pos_emb(pos_seq, batch_size=None)
        pos_emb = self.drop(pos_emb)

        core_out = self.drop(inputs)
        hiddens: List[torch.Tensor] = [core_out]

        if mlen > 0:
            attn_mask = torch.triu(torch.ones(seq_len, klen, device=device, dtype=torch.bool), diagonal=1 + mlen)
        else:
            attn_mask = torch.triu(torch.ones(seq_len, klen, device=device, dtype=torch.bool), diagonal=1)
        attn_mask = attn_mask.unsqueeze(-1)

        for layer_id, layer in enumerate(self.layers):
            mem_layer = mems[layer_id] if mems is not None else None
            core_out = layer(core_out, pos_emb, self.r_w_bias, self.r_r_bias, attn_mask, mem_layer)
            hiddens.append(core_out)

        output = self.final_norm(core_out)
        new_memory = self._update_memory(hiddens, mems, seq_len)
        return output, {"hidden_states": new_memory}

    def _update_memory(
        self,
        hiddens: List[torch.Tensor],
        mems: List[torch.Tensor],
        qlen: int,
    ) -> Optional[List[torch.Tensor]]:
        if self.memory_len <= 0:
            return None

        new_mems: List[torch.Tensor] = []
        with torch.no_grad():
            end_idx = mems[0].size(0) + qlen
            beg_idx = max(0, end_idx - self.memory_len)
            for layer_idx in range(1, len(hiddens)):
                prev_mem = mems[layer_idx - 1]
                cat = torch.cat([prev_mem, hiddens[layer_idx]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())
        return new_mems

    def initialize_memory(self, batch_size: int) -> Dict[str, Optional[List[torch.Tensor]]]:
        if self.memory_len <= 0:
            return {"hidden_states": None}
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        empty = torch.zeros(0, batch_size, self.d_model, device=device, dtype=dtype)
        return {"hidden_states": [empty.clone() for _ in range(self.n_layers)]}
