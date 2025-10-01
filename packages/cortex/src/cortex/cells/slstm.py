from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict

from cortex.cells.base import MemoryCell
from cortex.cells.conv import CausalConv1d

# Reuse utilities from mLSTM for normalization and init
from cortex.cells.mlstm import MultiHeadLayerNorm, bias_linspace_init_
from cortex.cells.registry import register_cell
from cortex.config import CausalConv1dConfig, sLSTMCellConfig
from cortex.kernels.pytorch.slstm import slstm_sequence_pytorch, slstm_sequence_triton
from cortex.types import MaybeState, ResetMask, Tensor
from cortex.utils import select_backend


class _HeadwiseLinearExpand(nn.Module):
    """Per-head linear layer with block-diagonal weights.

    Matches the reference LinearHeadwiseExpand with expand_factor_up=1.
    Weight shape: [NH, DH, DH]
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
    """Stateful sLSTM cell (vanilla PyTorch backend).

    - Input x: [B, T, H] or [B, H]
    - Output y: same shape as input

    State structure (TensorDict keys):
    - "y" [B, H]: last raw output before dropout/normalization. This is the
      recurrent signal used to compute the next step's recurrent preactivations (Ry).
    - "c" [B, H]: content accumulator; weighted running sum of tanh(z).
      Update: c_t = f_t * c_{t-1} + i_t * tanh(z_t).
    - "n" [B, H]: normalization accumulator; running sum of gate weights.
      Update: n_t = f_t * n_{t-1} + i_t. The exposed output divides c_t by n_t
      (after output gate) to stabilize scale across time.
    - "m" [B, H]: log-scale stabilizer for numerically stable i/f gating.
      Update: m_t = max(i_raw, m_{t-1} + logsigmoid(f_raw)); for the first step
      (all n==0) m_t is set to i_raw.
    - "conv" [B, KS, H]: optional causal-conv ring buffer present only when
      conv1d_kernel_size > 0; supplies the pre-activation path for i,f gates.

    Semantics:
    - Returned state always corresponds to the final timestep for the given call
      (both step and sequence modes).
    - Forward returns dropout + MultiHeadLayerNorm applied to "y"; the stored
      "y" in state is pre-dropout/normalization for correct recurrence.

    Layout relative to PostUpBlock:
    - LN -> sLSTMCell (optional conv -> gate projections -> sLSTM core -> dropout -> MH-LN)
      -> residual, then PostUpBlock applies its own FFN sublayer with pre-LN and residual.
    """

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

        # Gate projections (per-head structured like reference)
        H = cfg.hidden_size
        self.fgate = _HeadwiseLinearExpand(H, cfg.num_heads, bias=False)
        self.igate = _HeadwiseLinearExpand(H, cfg.num_heads, bias=False)
        self.zgate = _HeadwiseLinearExpand(H, cfg.num_heads, bias=False)
        self.ogate = _HeadwiseLinearExpand(H, cfg.num_heads, bias=False)

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
        """Forward pass through sLSTM cell.

        Dispatches to either Triton or PyTorch kernel based on device and shape constraints.
        """
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

        # Dispatch to appropriate kernel
        # Use Triton on CUDA when not in step mode and head_dim is power of 2
        allow_triton = not is_step and ((self.head_dim & (self.head_dim - 1)) == 0)

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
        return state


__all__ = ["sLSTMCell"]
