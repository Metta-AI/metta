"""Transformer-XL style attention cell with rolling memory.

Optionally replaces Q/K/V linear projections with AxonLayer when enabled via
configuration. Axon-backed layers maintain substates in the parent TensorDict
under the group ``xl_qkv``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from cortex.cells.base import MemoryCell
from cortex.cells.core import AxonLayer, update_parent_state
from cortex.cells.registry import register_cell
from cortex.config import XLCellConfig
from cortex.kernels.pytorch.txl import txl_pytorch
from cortex.types import MaybeState, ResetMask, Tensor


@register_cell(XLCellConfig)
class XLCell(MemoryCell):
    """Transformer-XL style multi-head attention with rolling memory."""

    def __init__(self, cfg: XLCellConfig) -> None:
        if cfg.hidden_size is None:
            raise ValueError("XLCellConfig.hidden_size must be specified")

        super().__init__(hidden_size=cfg.hidden_size)
        self.cfg = cfg

        d_model = cfg.hidden_size
        n_heads = cfg.n_heads
        head_dim = cfg.head_dim

        if n_heads <= 0:
            raise ValueError("n_heads must be > 0")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = head_dim if head_dim is not None else d_model // n_heads
        if self.d_head <= 0:
            raise ValueError("head_dim must be > 0")
        if head_dim is None and self.d_head * n_heads != d_model:
            raise ValueError("d_model must be divisible by n_heads when head_dim is not provided")

        self.mem_len = int(cfg.mem_len)
        # No chunking parameter; always process full provided sequence.

        proj_in = d_model
        proj_out = self.d_head * self.n_heads
        use_bias = cfg.use_bias

        if getattr(cfg, "use_axon_qkv", False):
            ax_cfg = getattr(cfg, "axon_qkv_config", None)
            self.q_proj = AxonLayer(proj_in, proj_out, cfg=ax_cfg, name="q", group="xl_qkv")
            self.k_proj = AxonLayer(proj_in, proj_out, cfg=ax_cfg, name="k", group="xl_qkv")
            self.v_proj = AxonLayer(proj_in, proj_out, cfg=ax_cfg, name="v", group="xl_qkv")
        else:
            self.q_proj = nn.Linear(proj_in, proj_out, bias=use_bias)
            self.k_proj = nn.Linear(proj_in, proj_out, bias=use_bias)
            self.v_proj = nn.Linear(proj_in, proj_out, bias=use_bias)
        self.r_proj = nn.Linear(proj_in, proj_out, bias=False)
        self.o_proj = nn.Linear(proj_out, d_model, bias=use_bias)

        self.u = nn.Parameter(torch.Tensor(self.n_heads, self.d_head))
        self.v = nn.Parameter(torch.Tensor(self.n_heads, self.d_head))

        self.attn_drop = nn.Dropout(cfg.attn_dropout)
        self.out_drop = nn.Dropout(cfg.out_dropout)
        self.scale = self.d_head**-0.5

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if isinstance(self.q_proj, nn.Linear):
            nn.init.xavier_uniform_(self.q_proj.weight)
        if isinstance(self.k_proj, nn.Linear):
            nn.init.xavier_uniform_(self.k_proj.weight)
        if isinstance(self.v_proj, nn.Linear):
            nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.r_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        if isinstance(self.q_proj, nn.Linear) and self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
        if isinstance(self.k_proj, nn.Linear) and self.k_proj.bias is not None:
            nn.init.zeros_(self.k_proj.bias)
        if isinstance(self.v_proj, nn.Linear) and self.v_proj.bias is not None:
            nn.init.zeros_(self.v_proj.bias)
        if self.o_proj.bias is not None:
            nn.init.zeros_(self.o_proj.bias)
        nn.init.zeros_(self.u)
        nn.init.zeros_(self.v)

    def init_state(self, batch: int, *, device: torch.device | str, dtype: torch.dtype) -> TensorDict:
        mem = torch.zeros(batch, 0, self.d_model, device=device, dtype=dtype)
        mem_seg = torch.zeros(batch, 0, device=device, dtype=torch.long)
        return TensorDict({"mem": mem, "mem_seg": mem_seg}, batch_size=[batch])

    def forward(
        self,
        x: Tensor,
        state: MaybeState,
        *,
        resets: Optional[ResetMask] = None,
    ) -> Tuple[Tensor, MaybeState]:
        is_step = x.dim() == 2
        if is_step:
            x_seq = x.unsqueeze(1)
        else:
            x_seq = x

        B, T, Hm = x_seq.shape
        if Hm != self.d_model:
            raise ValueError(f"Expected last dim {self.d_model}, got {Hm}")

        device = x_seq.device
        dtype = x_seq.dtype

        if state is None or "mem" not in state.keys() or "mem_seg" not in state.keys():
            mem_td = self.init_state(batch=B, device=device, dtype=dtype)
        else:
            mem_td = state

        mem = mem_td.get("mem")
        mem_seg = mem_td.get("mem_seg")
        if mem is None or mem_seg is None or mem.shape[0] != B:
            mem = torch.zeros(B, 0, self.d_model, device=device, dtype=dtype)
            mem_seg = torch.zeros(B, 0, device=device, dtype=torch.long)
        else:
            if mem.size(1) > self.mem_len > 0:
                mem = mem[:, -self.mem_len :, :].detach()
                mem_seg = mem_seg[:, -self.mem_len :]

        resets_bt = self._normalize_resets(resets, B, T, device)
        y_seq, new_mem, new_mem_seg = self._forward_block(
            x_seq,
            mem,
            mem_seg,
            resets_bt,
            parent_state=mem_td,
        )

        new_state = TensorDict({"mem": new_mem, "mem_seg": new_mem_seg}, batch_size=[B])
        # Preserve AxonLayer substates written into mem_td
        update_parent_state(new_state, mem_td)
        y_out: Tensor = y_seq.squeeze(1) if is_step else y_seq
        return y_out, new_state

    def reset_state(self, state: MaybeState, mask: ResetMask) -> MaybeState:
        if state is None:
            return None

        if "mem" not in state.keys() or "mem_seg" not in state.keys():
            return state

        mem = state.get("mem")
        mem_seg = state.get("mem_seg")
        if mem is None or mem_seg is None:
            return state

        mask_tensor = mask
        if mask_tensor.dtype != torch.bool:
            mask_tensor = mask_tensor != 0
        if mask_tensor.dim() > 1:
            mask_tensor = mask_tensor.any(dim=-1)
        mask_tensor = mask_tensor.view(-1)
        if mask_tensor.shape[0] != mem.shape[0]:
            raise ValueError(
                f"Reset mask batch {mask_tensor.shape[0]} does not match state batch {mem.shape[0]}",
            )

        if not mask_tensor.any():
            return state

        mem_new = mem.clone()
        mem_seg_new = mem_seg.clone()
        mem_new[mask_tensor] = 0.0
        mem_seg_new[mask_tensor] = 0

        out_state = TensorDict({"mem": mem_new, "mem_seg": mem_seg_new}, batch_size=state.batch_size)

        # Reset AxonLayer substates if enabled
        if isinstance(self.q_proj, AxonLayer):
            self.q_proj.reset_state(mask, state)
        if isinstance(self.k_proj, AxonLayer):
            self.k_proj.reset_state(mask, state)
        if isinstance(self.v_proj, AxonLayer):
            self.v_proj.reset_state(mask, state)

        # Preserve any auxiliary substates present in input state
        update_parent_state(out_state, state)
        return out_state

    def _forward_block(
        self,
        x_seq: torch.Tensor,
        mem: torch.Tensor,
        mem_seg: torch.Tensor,
        resets_bt: torch.Tensor,
        *,
        parent_state: TensorDict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x_seq.shape
        device = x_seq.device
        dtype = x_seq.dtype

        M = mem.size(1)
        if M > 0:
            last_seg_id = mem_seg[:, -1:].to(torch.long)
        else:
            last_seg_id = torch.zeros(B, 1, device=device, dtype=torch.long)
        seg_cur = last_seg_id + torch.cumsum(resets_bt, dim=1)
        seg_full = torch.cat([mem_seg.to(torch.long), seg_cur], dim=1)

        x_kv = torch.cat([mem, x_seq], dim=1) if M > 0 else x_seq
        L = x_kv.size(1)

        if isinstance(self.q_proj, AxonLayer):
            # For Axon-backed projections, ensure resets precisely follow segment
            # boundaries across memory + current tokens. Build a length-L mask
            # directly from seg_full so K/V streaming respects historical
            # boundaries instead of treating memory as a single unbroken span.
            q_lin = self.q_proj(x_seq, state=parent_state, resets=resets_bt)

            # Derive K/V resets from segment ids: reset whenever the segment id
            # changes between consecutive positions. Also force a reset at the
            # first position of the concatenated sequence to avoid leaking the
            # carried K/V Axon state from prior chunks into the replayed memory.
            dtype_mask = resets_bt.dtype if resets_bt is not None else torch.long
            resets_kv = torch.zeros(B, L, device=x_seq.device, dtype=dtype_mask)
            if L > 0:
                if L > 1:
                    changes = seg_full[:, 1:] != seg_full[:, :-1]
                    resets_kv[:, 1:] = changes.to(dtype_mask)
                resets_kv[:, 0] = 1  # start-of-mem boundary for K/V replay

            k_lin = self.k_proj(x_kv, state=parent_state, resets=resets_kv)
            v_lin = self.v_proj(x_kv, state=parent_state, resets=resets_kv)
        else:
            q_lin = self.q_proj(x_seq)
            k_lin = self.k_proj(x_kv)
            v_lin = self.v_proj(x_kv)

        q = self._shape_q(q_lin)
        k = self._shape_kv(k_lin)
        v = self._shape_kv(v_lin)

        r = self._relative_positions(L, device, dtype)
        r = self.r_proj(r)
        r = r.view(L, self.n_heads, self.d_head).permute(1, 0, 2).contiguous()
        ctx = txl_pytorch(q, k, v, r, seg_full, self.u, self.v, M, self.scale)

        y = ctx.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_head)
        y = self.o_proj(y)
        y = self.out_drop(y)

        if self.mem_len > 0:
            new_mem = torch.cat([mem, x_seq], dim=1) if M > 0 else x_seq
            new_mem_seg = torch.cat([mem_seg, seg_cur], dim=1)
            if new_mem.size(1) > self.mem_len:
                new_mem = new_mem[:, -self.mem_len :, :].detach()
                new_mem_seg = new_mem_seg[:, -self.mem_len :]
            else:
                new_mem = new_mem.detach()
        else:
            new_mem = torch.zeros(B, 0, self.d_model, device=device, dtype=dtype)
            new_mem_seg = torch.zeros(B, 0, device=device, dtype=torch.long)

        return y, new_mem, new_mem_seg

    def _normalize_resets(
        self,
        resets: Optional[ResetMask],
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if resets is None:
            return torch.zeros(batch_size, seq_len, device=device, dtype=torch.long)

        resets_bt = resets.to(device=device)
        if resets_bt.dtype == torch.bool:
            resets_bt = resets_bt.long()
        elif resets_bt.dtype.is_floating_point:
            resets_bt = (resets_bt > 0).long()
        else:
            resets_bt = resets_bt.long()

        if resets_bt.shape == (batch_size,) and seq_len == 1:
            resets_bt = resets_bt.view(batch_size, 1)

        if resets_bt.shape != (batch_size, seq_len):
            raise ValueError(f"resets tensor must have shape {(batch_size, seq_len)}, got {resets_bt.shape}")

        return resets_bt

    def _shape_q(self, tensor: torch.Tensor) -> torch.Tensor:
        B, T, _ = tensor.shape
        return tensor.view(B, T, self.n_heads, self.d_head).transpose(1, 2).contiguous()

    def _shape_kv(self, tensor: torch.Tensor) -> torch.Tensor:
        B, L, _ = tensor.shape
        return tensor.view(B, L, self.n_heads, self.d_head).transpose(1, 2).contiguous()

    def _relative_positions(
        self,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        d_model = self.d_model
        pos_seq = torch.arange(length - 1, -1, -1, device=device, dtype=dtype)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2, device=device, dtype=torch.float32) / d_model))
        sinusoid_inp = torch.outer(pos_seq.to(torch.float32), inv_freq)
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=1)
        if pos_emb.size(1) < d_model:
            pos_emb = F.pad(pos_emb, (0, 1), mode="constant", value=0.0)
        pos_emb = pos_emb[:, :d_model].to(dtype)
        return pos_emb


__all__ = ["XLCell"]
