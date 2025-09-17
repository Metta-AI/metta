# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Transformer-XL variant wrapping NVIDIA's reference implementation."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from metta.agent.modules.transformer_wrapper import TransformerWrapper
from metta.agent.pytorch.pytorch_agent_mixin import PyTorchAgentMixin

Tensor = torch.Tensor


def _add_and_scale(tensor1: Tensor, tensor2: Tensor, alpha: float) -> Tensor:
    """Helper that matches the fused addition + scaling used in NVIDIA's kernels."""
    return alpha * (tensor1 + tensor2)


class NvidiaPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding identical to the NVIDIA reference."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0.0, dim, 2.0) / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq: Tensor, batch_size: Optional[int] = None) -> Tensor:
        sinusoid_inp = torch.outer(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        if batch_size is not None:
            return pos_emb[:, None, :].expand(-1, batch_size, -1)
        return pos_emb[:, None, :]


class NvidiaPositionwiseFF(nn.Module):
    """Feed-forward layer from the NVIDIA Transformer-XL implementation."""

    def __init__(self, d_model: int, d_inner: int, dropout: float, pre_lnorm: bool) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        self.pre_lnorm = pre_lnorm

        self.core = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs: Tensor) -> Tensor:
        if self.pre_lnorm:
            core_out = self.core(self.layer_norm(inputs))
            return core_out + inputs

        core_out = self.core(inputs)
        return self.layer_norm(inputs + core_out)


class NvidiaRelPartialLearnableMultiHeadAttn(nn.Module):
    """Relative multi-head attention with learnable biases."""

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
        self.dropout = dropout
        self.dropatt = dropatt
        self.pre_lnorm = pre_lnorm

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        self.r_net = nn.Linear(d_model, n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt_layer = nn.Dropout(dropatt)
        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1.0 / math.sqrt(d_head)

    @staticmethod
    def _rel_shift(x: Tensor) -> Tensor:
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)
        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])
        return x_padded[1:].view_as(x)

    def forward(
        self,
        content: Tensor,
        rel_pos: Tensor,
        r_w_bias: Tensor,
        r_r_bias: Tensor,
        attn_mask: Optional[Tensor] = None,
        mems: Optional[Tensor] = None,
    ) -> Tensor:
        qlen, batch_size = content.size(0), content.size(1)

        if mems is not None and mems.numel() > 0:
            cat = torch.cat([mems, content], dim=0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(content))
            else:
                w_heads = self.qkv_net(content)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

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
        AC = torch.einsum("ibnd,jbnd->bnij", rw_head_q, w_head_k)

        rr_head_q = w_head_q + r_r_bias[None]
        BD = torch.einsum("ibnd,jnd->bnij", rr_head_q, r_head_k)
        BD = self._rel_shift(BD)

        attn_score = _add_and_scale(AC, BD, self.scale)

        if attn_mask is not None and attn_mask.any():
            if attn_mask.dim() == 2:
                attn_score = (
                    attn_score.float().masked_fill(attn_mask[None, None, :, :], float("-inf")).type_as(attn_score)
                )
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(attn_mask[:, None, :, :], float("-inf")).type_as(attn_score)

        attn_prob = F.softmax(attn_score, dim=3)
        attn_prob = self.dropatt_layer(attn_prob)

        attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, w_head_v)
        attn_vec = attn_vec.contiguous().view(qlen, batch_size, self.n_head * self.d_head)

        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            return content + attn_out
        return self.layer_norm(content + attn_out)


class NvidiaRelPartialLearnableDecoderLayer(nn.Module):
    """Decoder block from NVIDIA's Transformer-XL implementation."""

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
        content: Tensor,
        rel_pos: Tensor,
        r_w_bias: Tensor,
        r_r_bias: Tensor,
        attn_mask: Optional[Tensor],
        mems: Optional[Tensor],
    ) -> Tensor:
        output = self.attn(content, rel_pos, r_w_bias, r_r_bias, attn_mask, mems)
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
        self.memory_len = mem_len
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
        self.final_norm = nn.LayerNorm(d_model)

        nn.init.normal_(self.r_w_bias, mean=0.0, std=0.02)
        nn.init.normal_(self.r_r_bias, mean=0.0, std=0.02)

    def forward(
        self,
        inputs: Tensor,
        memory: Optional[List[Tensor]],
    ) -> Tuple[Tensor, List[Tensor]]:
        if inputs.dim() != 3:
            raise ValueError(f"Expected tensor of shape (T, B, D), received {inputs.shape}")

        seq_len, batch_size, _ = inputs.shape
        device = inputs.device
        dtype = inputs.dtype

        if memory is None or len(memory) != self.n_layers:
            mems: List[Tensor] = [
                torch.zeros(0, batch_size, self.d_model, device=device, dtype=dtype) for _ in range(self.n_layers)
            ]
        else:
            mems = [m.to(device) for m in memory]
            if mems and mems[0].size(1) != batch_size:
                mems = [m[:, :batch_size].contiguous() for m in mems]

        mlen = mems[0].size(0) if mems else 0
        klen = mlen + seq_len

        pos_seq = torch.arange(klen - 1, -1, -1.0, device=device, dtype=dtype)
        if self.clamp_len > 0:
            pos_seq = pos_seq.clamp(max=float(self.clamp_len))
        pos_emb = self.pos_emb(pos_seq)
        pos_emb = self.drop(pos_emb)

        core_out = self.drop(inputs)
        layer_outputs: List[Tensor] = []

        if mlen > 0:
            attn_mask = torch.triu(torch.ones(seq_len, klen, device=device, dtype=torch.bool), diagonal=1 + mlen)
        else:
            attn_mask = torch.triu(torch.ones(seq_len, klen, device=device, dtype=torch.bool), diagonal=1)

        for layer_idx, layer in enumerate(self.layers):
            mem_layer = mems[layer_idx] if mems else None
            core_out = layer(core_out, pos_emb, self.r_w_bias, self.r_r_bias, attn_mask, mem_layer)
            layer_outputs.append(core_out.detach())

        output = self.final_norm(core_out)
        new_memory = self._update_memory(layer_outputs, mems, seq_len)
        return output, new_memory

    def _update_memory(self, hids: List[Tensor], mems: List[Tensor], qlen: int) -> List[Tensor]:
        if self.mem_len <= 0 or not mems:
            return []

        new_mems: List[Tensor] = []
        with torch.no_grad():
            for hid, mem in zip(hids, mems, strict=False):
                if mem.size(0) == 0:
                    cat = hid
                else:
                    cat = torch.cat([mem, hid], dim=0)
                new_mems.append(cat[-self.mem_len :].detach())
        return new_mems

    def initialize_memory(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> List[Tensor]:
        if self.mem_len <= 0:
            return []
        empty = torch.zeros(0, batch_size, self.d_model, device=device, dtype=dtype)
        return [empty.clone() for _ in range(self.n_layers)]


class TransformerNvidiaPolicy(nn.Module):
    """Metta policy that uses NVIDIA's Transformer-XL blocks for sequence modeling."""

    def __init__(
        self,
        env,
        input_size: int = 128,
        hidden_size: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 512,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        memory_len: int = 32,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False
        self.action_space = env.single_action_space
        self.out_width = env.obs_width if hasattr(env, "obs_width") else 11
        self.out_height = env.obs_height if hasattr(env, "obs_height") else 11
        self.num_layers = max(env.feature_normalizations.keys()) + 1 if hasattr(env, "feature_normalizations") else 25

        self.cnn1 = nn.Conv2d(self.num_layers, 64, 5, 3)
        self.cnn2 = nn.Conv2d(64, 128, 3, 1)

        with torch.no_grad():
            test_output = self.cnn2(self.cnn1(torch.zeros(1, self.num_layers, self.out_width, self.out_height)))
            self.flattened_size = test_output.numel() // test_output.shape[0]

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.encoded_obs = nn.Linear(256, hidden_size)

        nn.init.orthogonal_(self.cnn1.weight, 1.0)
        nn.init.zeros_(self.cnn1.bias)
        nn.init.orthogonal_(self.cnn2.weight, 1.0)
        nn.init.zeros_(self.cnn2.bias)
        nn.init.orthogonal_(self.fc1.weight, math.sqrt(2))
        nn.init.zeros_(self.fc1.bias)
        nn.init.orthogonal_(self.encoded_obs.weight, math.sqrt(2))
        nn.init.zeros_(self.encoded_obs.bias)

        self.core = NvidiaTransformerCore(
            d_model=hidden_size,
            n_layers=n_layers,
            n_heads=n_heads,
            d_inner=d_ff,
            mem_len=memory_len,
            dropout=dropout,
            dropatt=dropout,
            pre_lnorm=True,
            clamp_len=max_seq_len,
        )
        self._transformer = self.core

        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 100),
        )
        for head in [self.critic, self.actor]:
            for layer in head:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, 0.01 if layer is head[-1] else math.sqrt(2))
                    nn.init.zeros_(layer.bias)

        max_values = [1.0] * self.num_layers
        if hasattr(env, "feature_normalizations"):
            for fid, norm in env.feature_normalizations.items():
                if fid < self.num_layers:
                    max_values[fid] = norm if norm > 0 else 1.0
        self.register_buffer("max_vec", torch.tensor(max_values, dtype=torch.float32)[None, :, None, None])
        self.active_action_names: List[str] = []
        self.num_active_actions = 100

    def initialize_to_environment(self, full_action_names: List[str], device: torch.device) -> None:
        self.active_action_names = full_action_names
        self.num_active_actions = len(full_action_names)

    def network_forward(self, inputs: Tensor) -> Tensor:
        x = inputs / self.max_vec
        x = F.relu(self.cnn2(F.relu(self.cnn1(x))))
        x = F.relu(self.fc1(self.flatten(x)))
        x = self.encoded_obs(x)
        return F.relu(x)

    def encode_observations(self, observations: Tensor, state: Optional[dict] = None) -> Tensor:
        if observations.dim() == 4:
            batch_size = observations.shape[0] * observations.shape[1]
            observations = observations.view(batch_size, *observations.shape[2:])
        else:
            batch_size = observations.shape[0]

        if observations.shape[-1] != 3:
            raise ValueError(f"Expected 3-channel token observations, got {observations.shape}")

        coords_byte = observations[..., 0].to(torch.uint8)
        x_coord_indices = ((coords_byte >> 4) & 0x0F).long()
        y_coord_indices = (coords_byte & 0x0F).long()
        attr_indices = observations[..., 1].long()
        attr_values = observations[..., 2].float()

        valid_tokens = coords_byte != 0xFF
        valid_attr = attr_indices < self.num_layers
        valid_mask = valid_tokens & valid_attr

        if (valid_tokens & ~valid_attr).any():
            attr_max = int(attr_indices.max().item())
            raise ValueError(f"Received attribute index >= {self.num_layers}: {attr_max}")

        dim_per_layer = self.out_width * self.out_height
        combined_index = attr_indices * dim_per_layer + x_coord_indices * self.out_height + y_coord_indices
        safe_index = torch.where(valid_mask, combined_index, torch.zeros_like(combined_index))
        safe_values = torch.where(valid_mask, attr_values, torch.zeros_like(attr_values))

        box_flat = torch.zeros(
            (batch_size, self.num_layers * dim_per_layer), dtype=attr_values.dtype, device=observations.device
        )
        box_flat.scatter_(1, safe_index, safe_values)

        box_obs = box_flat.view(batch_size, self.num_layers, self.out_width, self.out_height)
        return self.network_forward(box_obs)

    def decode_actions(self, hidden: Tensor, batch_size: int) -> Tuple[Tensor, Tensor]:
        values = self.critic(hidden).squeeze(-1)
        full_logits = self.actor(hidden)
        logits = full_logits[:, : self.num_active_actions]
        return logits, values

    def transformer(
        self,
        hidden: Tensor,
        terminations: Optional[Tensor] = None,
        memory: Optional[Dict[str, List[Tensor]]] = None,
    ) -> Tuple[Tensor, Dict[str, List[Tensor]]]:
        memory_list = None
        if memory is not None:
            hidden_states = memory.get("hidden_states")
            if hidden_states is None:
                raise ValueError("Transformer memory must include 'hidden_states'")
            memory_list = hidden_states

        output, new_memory = self.core(hidden, memory_list)

        if terminations is not None and new_memory:
            done_mask = terminations[-1].bool()
            if done_mask.any():
                for mem in new_memory:
                    mem[:, done_mask, :] = 0

        return output, {"hidden_states": new_memory}

    def initialize_memory(self, batch_size: int) -> Dict[str, List[Tensor]]:
        param = next(self.parameters())
        mem_list = self.core.initialize_memory(batch_size, device=param.device, dtype=param.dtype)
        return {"hidden_states": mem_list}


class TransformerNvidia(PyTorchAgentMixin, TransformerWrapper):
    """Agent wrapper around the NVIDIA Transformer-XL policy."""

    def __init__(
        self,
        env,
        policy: Optional[nn.Module] = None,
        input_size: int = 128,
        hidden_size: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 512,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        memory_len: int = 32,
        **kwargs,
    ) -> None:
        mixin_params = self.extract_mixin_params(kwargs)
        if policy is None:
            policy = TransformerNvidiaPolicy(
                env,
                input_size=input_size,
                hidden_size=hidden_size,
                n_heads=n_heads,
                n_layers=n_layers,
                d_ff=d_ff,
                max_seq_len=max_seq_len,
                dropout=dropout,
                memory_len=memory_len,
            )
        super().__init__(env, policy, hidden_size)
        self.init_mixin(**mixin_params)

    def forward(self, td: TensorDict, state: Optional[dict] = None, action: Optional[Tensor] = None) -> TensorDict:
        observations = td["env_obs"]

        if state is None:
            state = {"transformer_memory": None, "hidden": None}

        is_sequential = observations.dim() == 4
        if is_sequential:
            batch_size, seq_len = observations.shape[:2]
            flat_batch = batch_size * seq_len
            if td.batch_dims > 1:
                td = td.reshape(flat_batch)
        else:
            batch_size, seq_len = observations.shape[0], 1
            flat_batch = batch_size

        self.set_tensordict_fields(td, observations)

        memory_before = state.get("transformer_memory")
        hidden = self.policy.encode_observations(observations, state)

        if is_sequential:
            hidden = hidden.view(batch_size, seq_len, -1).transpose(0, 1)
        else:
            hidden = hidden.unsqueeze(0)

        terminations = state.get("terminations")
        if terminations is None:
            terminations = torch.zeros(seq_len, batch_size, device=hidden.device)

        hidden, new_memory = self.policy.transformer(hidden, terminations, state.get("transformer_memory"))
        normalized_memory = self._normalize_memory(new_memory)
        if normalized_memory is not None:
            state["transformer_memory"] = self._detach_memory(normalized_memory)
        else:
            state["transformer_memory"] = None

        segment_indices = td.get("_segment_indices", None)
        segment_pos = td.get("_segment_pos", None)
        if segment_indices is not None and segment_pos is not None:
            self._record_segment_memory(segment_indices, segment_pos, memory_before)

        if is_sequential:
            hidden = hidden.transpose(0, 1).contiguous().view(flat_batch, -1)
        else:
            hidden = hidden.squeeze(0)

        logits, values = self._decode_actions(hidden, flat_batch)

        if values.dim() > 1:
            values = values.squeeze(-1)

        if action is None:
            td = self.forward_inference(td, logits, values)
        else:
            td = self.forward_training(td, action, logits, values)

        return td
