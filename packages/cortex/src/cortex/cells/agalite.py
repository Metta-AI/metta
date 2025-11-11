"""AGaLiTe attention cell with recurrent discounted state."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from cortex.cells.base import MemoryCell
from cortex.cells.registry import register_cell
from cortex.config import AGaLiTeCellConfig
from cortex.types import MaybeState, ResetMask, Tensor
from cortex.utils import select_backend


@register_cell(AGaLiTeCellConfig)
class AGaLiTeCell(MemoryCell):
    """Feature-mapped attention with oscillatory basis and recurrent state."""

    def __init__(self, cfg: AGaLiTeCellConfig) -> None:
        if cfg.hidden_size is None:
            raise ValueError("AGaLiTeCellConfig.hidden_size must be specified")

        super().__init__(hidden_size=cfg.hidden_size)
        self.cfg = cfg

        H = cfg.hidden_size
        self.n_heads = int(cfg.n_heads)
        self.d_head = int(cfg.head_dim or (H // self.n_heads))
        if self.d_head * self.n_heads != H:
            raise ValueError("hidden_size must be divisible by n_heads (or specify head_dim)")

        self.eta = int(cfg.eta)
        self.r = int(cfg.r)
        self.eps = float(cfg.eps)
        self.drop = nn.Dropout(cfg.dropout)

        # Projections (match baseline JAX: single dense for K/Q/V/β/γ, single dense for p1/p2/p3)
        self.kqvbg_proj = nn.Linear(H, self.n_heads * self.d_head * 5, bias=False)
        self.p123_proj = nn.Linear(H, self.n_heads * self.eta * 3, bias=False)
        self.out_proj = nn.Linear(self.n_heads * self.d_head, H, bias=True)

        # Oscillation frequencies
        omegas = torch.linspace(-math.pi, math.pi, self.r)
        self.register_buffer("omegas", omegas)

        # Small cache for cos tables keyed by (T, device, dtype)
        self._cos_cache: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}

        # Feature dimension after mapping (eta * head_dim)
        self._feat_dim = self.eta * self.d_head
        # Initialize projections orthogonally to mirror baseline defaults
        with torch.no_grad():
            nn.init.orthogonal_(self.kqvbg_proj.weight, gain=math.sqrt(2))
            nn.init.orthogonal_(self.p123_proj.weight, gain=math.sqrt(2))
            nn.init.orthogonal_(self.out_proj.weight, gain=math.sqrt(2))
            if self.out_proj.bias is not None:
                nn.init.zeros_(self.out_proj.bias)

    # --------------------------- MemoryCell API ---------------------------
    def init_state(self, batch: int, *, device: torch.device | str, dtype: torch.dtype) -> TensorDict:
        B = int(batch)
        tilde_k = torch.zeros(B, self.r, self.n_heads, self._feat_dim, device=device, dtype=dtype)
        tilde_v = torch.zeros(B, self.r, self.n_heads, self.d_head, device=device, dtype=dtype)
        s = torch.zeros(B, self.n_heads, self._feat_dim, device=device, dtype=dtype)
        tick = torch.ones(B, device=device, dtype=dtype)
        return TensorDict({"tilde_k": tilde_k, "tilde_v": tilde_v, "s": s, "tick": tick}, batch_size=[B])

    def forward(
        self,
        x: Tensor,
        state: MaybeState,
        *,
        resets: Optional[ResetMask] = None,
    ) -> Tuple[Tensor, MaybeState]:
        # Normalize input shape to [B, T, H]
        is_step = x.dim() == 2
        if is_step:
            x_seq = x.unsqueeze(1)
        else:
            x_seq = x
        B, T, H = x_seq.shape
        if H != self.hidden_size:
            raise ValueError(f"Expected last dim {self.hidden_size}, got {H}")

        # Prepare/validate state
        if state is None or not all(k in state.keys() for k in ("tilde_k", "tilde_v", "s", "tick")):
            st = self.init_state(B, device=x_seq.device, dtype=x_seq.dtype)
        else:
            st = state

        # Normalize resets to [B, T] of 0/1 longs
        resets_bt = self._normalize_resets(resets, B, T, x_seq.device)

        # Joint projection then split to (k, q, v, beta, gamma) per head
        kqvbg = self.kqvbg_proj(x_seq).view(B, T, self.n_heads, 5 * self.d_head)
        k, q, v, beta, gamma = torch.split(kqvbg, self.d_head, dim=-1)
        beta = torch.sigmoid(beta)

        # Auxiliary projections p1/p2/p3 (per head, length eta)
        p123 = self.p123_proj(x_seq).view(B, T, self.n_heads, 3 * self.eta)
        p1, p2, p3 = torch.split(p123, self.eta, dim=-1)

        # Feature expansions using ReLU for keys/queries and sigmoid for gamma path
        # shapes: keys_feat, queries_feat, gammas_feat -> [B, T, Hh, eta*Dh]
        keys_feat = torch.einsum("bthd,bthe->bthde", F.relu(k), F.relu(p1)).reshape(B, T, self.n_heads, self._feat_dim)
        queries_feat = torch.einsum("bthd,bthe->bthde", F.relu(q), F.relu(p2)).reshape(
            B, T, self.n_heads, self._feat_dim
        )
        gammas_feat = torch.einsum("bthd,bthe->bthde", torch.sigmoid(gamma), torch.sigmoid(p3)).reshape(
            B, T, self.n_heads, self._feat_dim
        )

        # Time-major for kernel calls
        v_tbh = v.transpose(0, 1).contiguous()
        beta_tbh = beta.transpose(0, 1).contiguous()
        queries_t = queries_feat.transpose(0, 1).contiguous()  # [T,B,Hh,eta*Dh]
        keys_t = keys_feat.transpose(0, 1).contiguous()
        gammas_t = gammas_feat.transpose(0, 1).contiguous()

        # Oscillation terms using tick
        tilde_k_prev = st.get("tilde_k")
        tilde_v_prev = st.get("tilde_v")
        s_prev = st.get("s")
        tick = st.get("tick")
        assert tilde_k_prev is not None and tilde_v_prev is not None and s_prev is not None and tick is not None

        # Oscillation terms using cos((tick + t) * omega)
        cos_terms = self._cos_terms(T, tick, device=x_seq.device, dtype=x_seq.dtype)  # [T,B,r]

        # Gate values and combine oscillations
        gated_values = v_tbh * beta_tbh  # [T,B,Hh,Dh]
        gated_keys = keys_t * gammas_t  # [T,B,Hh,F]

        cos_expanded = cos_terms.unsqueeze(-1).unsqueeze(-1)  # [T,B,r,1,1]
        values_osc = gated_values.unsqueeze(2) * cos_expanded  # [T,B,r,Hh,Dh]
        keys_osc = gated_keys.unsqueeze(2) * cos_expanded  # [T,B,r,Hh,F]

        # Discounts; incorporate resets: when reset==1, discount multiplier is 0 for that step
        reset_mask = (1 - resets_bt.to(beta_tbh.dtype)).transpose(0, 1).unsqueeze(-1).unsqueeze(-1)
        discount_gamma = (1 - gammas_t) * reset_mask  # [T,B,Hh,F]
        discount_beta = (1 - beta_tbh) * reset_mask  # [T,B,Hh,Dh]

        discount_gamma_r = discount_gamma.unsqueeze(2).expand(-1, -1, self.r, -1, -1)
        discount_beta_r = discount_beta.unsqueeze(2).expand(-1, -1, self.r, -1, -1)

        # Select backend for discounted sum
        allow_cuda = x_seq.is_cuda
        pytorch_fn = "cortex.kernels.pytorch.agalite:discounted_sum_pytorch"
        cuda_fn = "cortex.kernels.cuda.agalite.discounted_sum_cuda:discounted_sum_cuda" if allow_cuda else None
        ds_fn = select_backend(
            triton_fn=None,
            pytorch_fn=pytorch_fn,
            tensor=x_seq,
            allow_triton=False,
            cuda_fn=cuda_fn,
            allow_cuda=allow_cuda,
        )

        final_keys = ds_fn(tilde_k_prev, keys_osc, discount_gamma_r)  # [T,B,r,Hh,F]
        final_values = ds_fn(tilde_v_prev, values_osc, discount_beta_r)  # [T,B,r,Hh,Dh]
        final_s = ds_fn(s_prev, gated_keys, discount_gamma)  # [T,B,Hh,F]

        # Attention composition
        keys_dot_queries = torch.einsum("tbrhD,tbhD->tbrh", final_keys, queries_t)
        kv = torch.einsum("tbrhd,tbrh->tbhd", final_values, keys_dot_queries)
        norm = torch.einsum("tbhD,tbhD->tbh", final_s, queries_t)
        attn_out = kv / (2 * self.r * norm.unsqueeze(-1) + self.eps)

        # Project out
        y = self.drop(self.out_proj(attn_out.reshape(T, B, self.n_heads * self.d_head)))
        y = y.transpose(0, 1).contiguous()  # [B,T,H]
        y = y.squeeze(1) if is_step else y

        # Update state with last time step
        new_tick = tick + T
        new_state = TensorDict(
            {
                "tilde_k": final_keys[-1].detach(),
                "tilde_v": final_values[-1].detach(),
                "s": final_s[-1].detach(),
                "tick": new_tick,
            },
            batch_size=[B],
        )

        return y, new_state

    def reset_state(self, state: MaybeState, mask: ResetMask) -> MaybeState:
        if state is None:
            return None
        tilde_k = state.get("tilde_k")
        tilde_v = state.get("tilde_v")
        s = state.get("s")
        tick = state.get("tick")
        if tilde_k is None or tilde_v is None or s is None or tick is None:
            return state
        B = tilde_k.shape[0]
        mask_b = mask.to(dtype=tilde_k.dtype).view(-1, 1, 1, 1)
        tilde_k = tilde_k * (1.0 - mask_b)
        tilde_v = tilde_v * (1.0 - mask_b)
        s = s * (1.0 - mask.view(-1, 1, 1).to(dtype=s.dtype))
        tick = tick * (1.0 - mask.to(dtype=tick.dtype))
        return TensorDict({"tilde_k": tilde_k, "tilde_v": tilde_v, "s": s, "tick": tick}, batch_size=[B])

    # ----------------------------- Internals -----------------------------
    def _cos_terms(self, timesteps: int, tick: torch.Tensor, *, device: torch.device, dtype: torch.dtype):
        steps = torch.arange(1, timesteps + 1, device=device, dtype=dtype).view(timesteps, 1)
        # cos((tick + t) * omega)
        angles = (tick.view(1, -1, 1).to(dtype) + steps.view(-1, 1, 1)) * self.omegas.view(1, 1, -1).to(
            device=device, dtype=dtype
        )
        return torch.cos(angles)  # [T,B,r]

    def _normalize_resets(
        self, resets: Optional[ResetMask], batch_size: int, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        if resets is None:
            return torch.zeros(batch_size, seq_len, device=device, dtype=torch.long)

        rb = (resets.to(device=device) != 0).to(dtype=torch.long)

        if rb.shape == (batch_size, seq_len):
            return rb
        if rb.shape == (batch_size,):
            # Broadcast per-batch mask across time
            return rb.view(batch_size, 1).expand(batch_size, seq_len)

        raise ValueError(f"resets must have shape {(batch_size, seq_len)} or {(batch_size,)}, got {tuple(rb.shape)}")


__all__ = ["AGaLiTeCell"]
