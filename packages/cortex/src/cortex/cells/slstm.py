"""Structured LSTM cell with per-head recurrence and stabilized gating."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict

from cortex.cells.base import MemoryCell
from cortex.cells.conv import CausalConv1d
from cortex.cells.core import AxonLayer

# Reuse utilities from mLSTM for normalization and init
from cortex.cells.mlstm import MultiHeadLayerNorm, bias_linspace_init_
from cortex.cells.registry import register_cell
from cortex.config import AxonsConfig, CausalConv1dConfig, sLSTMCellConfig
from cortex.kernels.pytorch.slstm import slstm_sequence_pytorch
from cortex.kernels.triton.slstm import slstm_sequence_triton
from cortex.types import MaybeState, ResetMask, Tensor
from cortex.utils import select_backend


class _HeadwiseLinearExpand(nn.Module):
    """Per-head linear layer with block-diagonal weight structure.

    This is the legacy behavior used when AxonLayer integration is disabled.
    """

    def __init__(self, in_features: int, num_heads: int, bias: bool = False) -> None:
        super().__init__()
        assert in_features % num_heads == 0, "in_features must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = in_features // num_heads

        self.weight = nn.Parameter(torch.empty(num_heads, self.head_dim, self.head_dim))
        self.bias = nn.Parameter(torch.empty(in_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # small init as used in reference components
        std = (2.0 / (5.0 * self.head_dim)) ** 0.5
        nn.init.normal_(self.weight, mean=0.0, std=std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H] or [B, T, H]
        if x.dim() == 2:
            B, H = x.shape
            xh = x.view(B, self.num_heads, self.head_dim)
            y = torch.einsum("bnd,ndf->bnf", xh, self.weight)
            y = y.reshape(B, H)
        else:
            B, T, H = x.shape
            xh = x.view(B, T, self.num_heads, self.head_dim)
            y = torch.einsum("btnd,ndf->btnf", xh, self.weight)
            y = y.reshape(B, T, H)
        if self.bias is not None:
            y = y + self.bias
        return y


@register_cell(sLSTMCellConfig)
class sLSTMCell(MemoryCell):
    """Structured LSTM cell with per-head gating, state normalization, and optional causal conv."""

    def __init__(self, cfg: sLSTMCellConfig) -> None:
        super().__init__(hidden_size=cfg.hidden_size)
        self.cfg = cfg

        assert cfg.hidden_size % cfg.num_heads == 0, "hidden_size must be divisible by num_heads"
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.hidden_size // cfg.num_heads

        # Optional depthwise causal conv preprocessing
        self.conv_kernel_size = cfg.conv1d_kernel_size
        if self.conv_kernel_size > 0:
            self.conv1d_cell = CausalConv1d(
                CausalConv1dConfig(
                    hidden_size=cfg.hidden_size,
                    kernel_size=self.conv_kernel_size,
                    causal_conv_bias=True,
                    channel_mixing=False,
                )
            )
            self.conv_act = nn.SiLU()
        else:
            self.conv1d_cell = None
            self.conv_act = None

        # Gate projections (choose Axons or legacy Linear based on flag)
        H = cfg.hidden_size
        NH = cfg.num_heads
        if cfg.use_axon_layer:
            # Fused Axon gates: compute [i,f] from x_conv and [z,o] from x_seq
            # with two AxonLayer calls H -> 2H to reduce per-chunk overhead.
            is_pow2 = (H & (H - 1)) == 0 and H > 0
            ax_cfg = AxonsConfig(
                hidden_size=H,
                out_dim=2 * H,
                out_rank=cfg.axon_rank,
                use_srht=bool(is_pow2),
                srht_permute=True,
            )
            self.if_fused = AxonLayer(H, 2 * H, cfg=ax_cfg, name="if_fused", group="slstm")
            self.zo_fused = AxonLayer(H, 2 * H, cfg=ax_cfg, name="zo_fused", group="slstm")
        else:
            self.fgate = _HeadwiseLinearExpand(H, NH, bias=False)
            self.igate = _HeadwiseLinearExpand(H, NH, bias=False)
            self.zgate = _HeadwiseLinearExpand(H, NH, bias=False)
            self.ogate = _HeadwiseLinearExpand(H, NH, bias=False)

        # Recurrent kernel (per-head, per-gate). Shape: [NH, 4*DH, DH]
        self.recurrent_kernel = nn.Parameter(torch.zeros(self.num_heads, 4 * self.head_dim, self.head_dim))
        # Bias per gate per head: [NH, 4, DH]
        self.bias = nn.Parameter(torch.zeros(self.num_heads, 4, self.head_dim))

        # Output normalization and dropout
        self.outnorm = MultiHeadLayerNorm(cfg.hidden_size, weight=True, bias=False)
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Keep recurrent kernel at 0 (identity-free start)
        nn.init.zeros_(self.recurrent_kernel)
        # Gate projections small init via submodules
        # Biases: forget gate positive, others zero
        for h in range(self.num_heads):
            # order: [i, f, z, o] -> index 1 is forget
            bias_linspace_init_(self.bias[h, 1], start=3.0, end=6.0)
            nn.init.zeros_(self.bias[h, 0])
            nn.init.zeros_(self.bias[h, 2])
            nn.init.zeros_(self.bias[h, 3])
        # Norm params
        self.outnorm.reset_parameters()
        # Conv params
        if self.conv1d_cell is not None:
            self.conv1d_cell.reset_parameters()

    def init_state(self, batch: int, *, device: torch.device | str, dtype: torch.dtype) -> TensorDict:
        B = batch
        H = self.cfg.hidden_size
        zero = torch.zeros(B, H, device=device, dtype=dtype)
        td = TensorDict({"y": zero.clone(), "c": zero.clone(), "n": zero.clone(), "m": zero.clone()}, batch_size=[B])
        if self.conv1d_cell is not None:
            conv_state = self.conv1d_cell.init_state(batch=B, device=device, dtype=dtype)
            td.update(conv_state)
        return td

    def _apply_conv(
        self, x_seq: Tensor, conv_state: MaybeState, resets: Optional[ResetMask]
    ) -> tuple[Tensor, MaybeState]:
        # Ensure step inputs use the conv cell's step path so the ring buffer state is updated correctly.
        if self.conv1d_cell is None:
            return x_seq, conv_state

        if x_seq.dim() == 3 and x_seq.shape[1] == 1:
            # Step mode: pass [B, H] to CausalConv1d so it updates its ring buffer
            y_step, new_conv_state = self.conv1d_cell(x_seq.squeeze(1), conv_state, resets=resets)
            y_step = self.conv_act(y_step)
            return y_step.unsqueeze(1), new_conv_state  # [B, 1, H]

        # Sequence mode: pass through as-is
        y, new_conv_state = self.conv1d_cell(x_seq, conv_state, resets=resets)
        return self.conv_act(y), new_conv_state  # type: ignore[arg-type]

    def _normalize_output(self, y_seq: Tensor) -> Tensor:
        # y_seq: [B, T, H] or [B, H]
        is_step = y_seq.dim() == 2
        if is_step:
            y = y_seq.view(y_seq.shape[0], self.num_heads, 1, self.head_dim)
            y = self.dropout(y)
            y = self.outnorm(y).view(y_seq.shape[0], -1)
            return y
        else:
            B, T, H = y_seq.shape
            y = y_seq.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            y = self.dropout(y)
            y = self.outnorm(y).transpose(1, 2).reshape(B, T, H)
            return y

    def forward(
        self,
        x: Tensor,
        state: MaybeState,
        *,
        resets: Optional[ResetMask] = None,
    ) -> Tuple[Tensor, MaybeState]:
        """Apply sLSTM recurrence with automatic backend selection."""
        # Handle [B, H] vs [B, T, H]
        is_step = x.dim() == 2
        if is_step:
            x_seq = x.unsqueeze(1)
        else:
            x_seq = x

        B, T, H = x_seq.shape
        NH, DH = self.num_heads, self.head_dim

        # Initialize state
        if state is None or not all(k in state for k in ("y", "c", "n", "m")):
            st = self.init_state(batch=B, device=x_seq.device, dtype=x_seq.dtype)
        else:
            st = state

        y_prev = st.get("y")
        c_prev = st.get("c")
        n_prev = st.get("n")
        m_prev = st.get("m")

        # Handle resets - prepare for kernel dispatch
        # For step mode, apply resets to initial states before kernel call
        # For sequence mode, pass resets to kernel to apply per-timestep
        kernel_resets: Optional[torch.Tensor] = None
        if resets is not None:
            if is_step:
                # Step mode: apply reset to initial states
                mask = resets.to(dtype=y_prev.dtype).view(B, 1)
                y_prev = y_prev * (1.0 - mask)
                c_prev = c_prev * (1.0 - mask)
                n_prev = n_prev * (1.0 - mask)
                m_prev = m_prev * (1.0 - mask)
            else:
                # Sequence mode: prepare resets for kernel (B, T)
                # resets could already be (B, T) or might need reshaping
                if resets.dim() == 1:
                    # If (B,) broadcast to (B, T)
                    kernel_resets = resets.unsqueeze(1).expand(B, T)
                else:
                    # Already (B, T)
                    kernel_resets = resets

        # Extract conv state dict (if present)
        conv_state_in: MaybeState
        if self.conv1d_cell is not None and st is not None and "conv" in st.keys():
            conv_state_in = TensorDict({"conv": st.get("conv")}, batch_size=[B])
        else:
            conv_state_in = None

        # Apply causal conv preprocessing
        x_conv, conv_state_new = self._apply_conv(x_seq, conv_state_in, resets=resets)

        # Compute gate preactivations
        if self.cfg.use_axon_layer:
            # Two fused Axon calls: [i,f] from x_conv, [z,o] from x_seq
            if_f = self.if_fused(x_conv, state=st, resets=resets)  # [B, T, 2H] or [B, 2H]
            z_o = self.zo_fused(x_seq, state=st, resets=resets)  # [B, T, 2H] or [B, 2H]
            i_pre, f_pre = torch.chunk(if_f, 2, dim=-1)
            z_pre, o_pre = torch.chunk(z_o, 2, dim=-1)
        else:
            i_pre = self.igate(x_conv)
            f_pre = self.fgate(x_conv)
            z_pre = self.zgate(x_seq)
            o_pre = self.ogate(x_seq)

        # Prepare inputs in unified format for kernel dispatch
        # Wx as (B, T, 4, NH, DH) with order (i, f, z, o)
        Wx_seq = torch.stack(
            (
                i_pre.view(B, T, NH, DH),
                f_pre.view(B, T, NH, DH),
                z_pre.view(B, T, NH, DH),
                o_pre.view(B, T, NH, DH),
            ),
            dim=2,
        )

        # Recurrent weights R: (4, NH, DH, DH) in order (i, f, z, o)
        R = self.recurrent_kernel.view(NH, 4, DH, DH).permute(1, 0, 2, 3).contiguous()
        # Bias: (4, NH, DH) in order (i, f, z, o)
        b = self.bias.permute(1, 0, 2).contiguous()

        # Initial states (h, c, n, m) as (4, B, NH, DH)
        y0 = y_prev.view(B, NH, DH)
        c0 = c_prev.view(B, NH, DH)
        n0 = n_prev.view(B, NH, DH)
        m0 = m_prev.view(B, NH, DH)
        states0 = torch.stack((y0, c0, n0, m0), dim=0)

        allow_triton = (self.head_dim & (self.head_dim - 1)) == 0
        logging.debug(f"head_dim: {self.head_dim}, allow_triton: {allow_triton}, is_step: {is_step}")
        backend_fn = select_backend(
            triton_fn=slstm_sequence_triton,
            pytorch_fn=slstm_sequence_pytorch,
            tensor=x_seq,
            allow_triton=allow_triton,
        )

        all_states, last_state = backend_fn(
            Wx=Wx_seq,
            R=R,
            b=b,
            initial_states=states0,
            resets=kernel_resets,
        )

        # Extract outputs from kernel results
        # all_states: (T, 4, B, NH, DH); last_state: (4, B, NH, DH)
        y_seq = all_states[:, 0].permute(1, 0, 2, 3).reshape(B, T, H)
        y_t = last_state[0].reshape(B, H)
        c_t = last_state[1].reshape(B, H)
        n_t = last_state[2].reshape(B, H)
        m_t = last_state[3].reshape(B, H)

        # Create new state
        new_state = TensorDict({"y": y_t, "c": c_t, "n": n_t, "m": m_t}, batch_size=[B])
        if conv_state_new is not None:
            new_state.update(conv_state_new)

        # Apply normalization and dropout
        y_out = self._normalize_output(y_seq)

        if is_step:
            return y_out.squeeze(1) if y_out.dim() == 3 else y_out, new_state
        else:
            return y_out, new_state

    def reset_state(self, state: MaybeState, mask: ResetMask) -> MaybeState:
        if state is None:
            return None
        # Broadcast mask to [B, 1]
        mask_b = mask.to(dtype=state["y"].dtype).view(-1, 1)
        for k in ("y", "c", "n", "m"):
            if k in state.keys():
                state[k] = state[k] * (1.0 - mask_b)

        # Reset conv buffer if present
        if self.conv1d_cell is not None and "conv" in state.keys():
            conv_td = TensorDict({"conv": state["conv"]}, batch_size=[state["conv"].shape[0]])
            conv_td = self.conv1d_cell.reset_state(conv_td, mask)
            if "conv" in conv_td:
                state["conv"] = conv_td["conv"]

        # Reset Axon gate substates (per-head) when enabled
        if self.cfg.use_axon_layer:
            self.if_fused.reset_state(mask, state)
            self.zo_fused.reset_state(mask, state)
        return state


__all__ = ["sLSTMCell"]
