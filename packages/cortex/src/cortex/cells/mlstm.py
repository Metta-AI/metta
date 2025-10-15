"""Matrix LSTM cell with parallel chunk processing and normalized state."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict

from cortex.cells.base import MemoryCell
from cortex.cells.conv import CausalConv1d
from cortex.cells.core import AxonLayer
from cortex.cells.registry import register_cell
from cortex.config import AxonsConfig, CausalConv1dConfig, mLSTMCellConfig
from cortex.kernels.pytorch.mlstm import (
    mlstm_chunkwise_simple,
    mlstm_recurrent_step_stabilized_simple,
)
from cortex.kernels.triton.mlstm import mlstm_chunkwise_triton
from cortex.types import MaybeState, ResetMask, Tensor
from cortex.utils import select_backend


def bias_linspace_init_(param: torch.Tensor, start: float = 3.4, end: float = 6.0) -> torch.Tensor:
    """Linearly spaced bias init across dimensions."""
    assert param.dim() == 1, f"param must be 1-dimensional (typically a bias), got {param.dim()}"
    n_dims = param.shape[0]
    init_vals = torch.linspace(start, end, n_dims)
    with torch.no_grad():
        param.copy_(init_vals)
    return param


class MultiHeadLayerNorm(nn.Module):
    """Multi-head layer normalization using group normalization."""

    def __init__(self, ndim: int, weight: bool = True, bias: bool = False, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim)) if weight else None
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps
        self.ndim = ndim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.dim() == 4, "Input must be 4D tensor (B, NH, S, DH)"
        B, NH, S, DH = input.shape

        gn_in_1 = input.transpose(1, 2)  # (B, S, NH, DH)
        gn_in_2 = gn_in_1.reshape(B * S, NH * DH)  # (B * S, NH * DH)
        out = torch.nn.functional.group_norm(
            gn_in_2,
            num_groups=NH,
            weight=self.weight,
            bias=self.bias,
            eps=self.eps,
        )
        # (B * S), (NH * DH) -> (B, S, NH, DH) -> (B, NH, S, DH)
        out = out.view(B, S, NH, DH).transpose(1, 2)
        return out

    def reset_parameters(self):
        if self.weight is not None:
            nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


@register_cell(mLSTMCellConfig)
class mLSTMCell(MemoryCell):
    """Matrix LSTM cell with matrix-valued state and parallel/recurrent processing modes."""

    def __init__(self, cfg: mLSTMCellConfig) -> None:
        super().__init__(hidden_size=cfg.hidden_size)
        self.cfg = cfg

        assert cfg.hidden_size % cfg.num_heads == 0, "hidden_size must be divisible by num_heads"
        self.head_dim = cfg.hidden_size // cfg.num_heads

        # Input/forget gates: either Axon-backed or legacy Linear depending on flag
        H = cfg.hidden_size
        NH = cfg.num_heads
        if cfg.use_axon_layer:
            in_features = 3 * H
            out_features = NH
            is_pow2 = (in_features & (in_features - 1)) == 0 and in_features > 0
            ax_cfg = AxonsConfig(use_untraced_linear=True)
            self.igate = AxonLayer(
                in_features,
                out_features,
                cfg=ax_cfg,
                name="igate",
                group="mlstm",
            )
            self.fgate = AxonLayer(in_features, out_features, cfg=ax_cfg, name="fgate", group="mlstm")
        else:
            self.igate = nn.Linear(3 * H, NH)
            self.fgate = nn.Linear(3 * H, NH)

        # Q/K/V preprocessing: always apply causal conv; optionally add Axon QKV
        self.conv_kernel_size = cfg.conv1d_kernel_size
        conv_config = CausalConv1dConfig(
            hidden_size=cfg.hidden_size,
            kernel_size=self.conv_kernel_size,
            causal_conv_bias=True,
            channel_mixing=False,  # depthwise convolution
        )
        self.conv1d_cell = CausalConv1d(conv_config)
        self.conv_act = nn.SiLU()

        if not cfg.use_axon_qkv:
            self.qkv_act = None
            self.q_layer = None
            self.k_layer = None
            self.v_layer = None
            self.qk_layer = None
        else:
            H = int(cfg.hidden_size)
            out_rank = cfg.axon_rank
            is_pow2 = (H & (H - 1)) == 0 and H > 0
            qkv_cfg = AxonsConfig(
                hidden_size=H, out_dim=H, out_rank=out_rank, use_srht=bool(is_pow2), srht_permute=True
            )
            self.qkv_act = nn.SiLU()  # match conv+SiLU behavior
            # Shared-QK: single layer feeds both q and k; v has its own layer
            self.qk_layer = AxonLayer(H, H, cfg=qkv_cfg, name="qk", group="mlstm_qkv")
            self.v_layer = AxonLayer(H, H, cfg=qkv_cfg, name="v", group="mlstm_qkv")
            self.q_layer = None
            self.k_layer = None

        if not cfg.use_axon_qkv:
            self.qkv_act = None
            self.q_layer = None
            self.k_layer = None
            self.v_layer = None
            self.qk_layer = None
        else:
            H = int(cfg.hidden_size)
            is_pow2 = (H & (H - 1)) == 0 and H > 0
            qkv_cfg = AxonsConfig(hidden_size=H, out_dim=H,use_untraced_linear=True)
            self.qkv_act = nn.SiLU()  # match conv+SiLU behavior
            # Shared-QK: single layer feeds both q and k; v has its own layer
            self.qk_layer = AxonLayer(H, H, cfg=qkv_cfg, name="qk", group="mlstm_qkv")
            self.v_layer = AxonLayer(H, H, cfg=qkv_cfg, name="v", group="mlstm_qkv")
            self.q_layer = None
            self.k_layer = None

        # Output normalization
        self.outnorm = MultiHeadLayerNorm(cfg.hidden_size, weight=True, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        self.outnorm.reset_parameters()
        if self.conv1d_cell is not None:
            self.conv1d_cell.reset_parameters()
        # Initialize gates
        if not self.cfg.use_axon_layer:
            # Forget gate initialization (encourages retention)
            torch.nn.init.zeros_(self.fgate.weight)
            bias_linspace_init_(self.fgate.bias, start=3.0, end=6.0)
            # Input gate initialization
            torch.nn.init.zeros_(self.igate.weight)
            torch.nn.init.normal_(self.igate.bias, mean=0.0, std=0.1)
        else:
            # When using Axon-backed gates, initialize the internal linear branch
            f_lin = getattr(self.fgate, "linear", None)
            if f_lin is not None:
                torch.nn.init.zeros_(f_lin.weight)
                if f_lin.bias is not None:
                    bias_linspace_init_(f_lin.bias, start=3.0, end=6.0)
            i_lin = getattr(self.igate, "linear", None)
            if i_lin is not None:
                torch.nn.init.zeros_(i_lin.weight)
                if i_lin.bias is not None:
                    torch.nn.init.normal_(i_lin.bias, mean=0.0, std=0.1)

    def init_state(self, batch: int, *, device: torch.device | str, dtype: torch.dtype) -> TensorDict:
        """Initialize state tensors."""
        B = batch
        NH = self.cfg.num_heads
        DH = self.head_dim

        c_state = torch.zeros(B, NH, DH, DH, device=device, dtype=dtype)
        n_state = torch.zeros(B, NH, DH, 1, device=device, dtype=dtype)
        m_state = torch.zeros(B, NH, 1, 1, device=device, dtype=dtype)

        # Get conv state from the conv1d cell if in use
        if self.conv1d_cell is not None:
            conv_state = self.conv1d_cell.init_state(batch, device=device, dtype=dtype)
        else:
            conv_state = TensorDict({}, batch_size=[B])

        # Combine all states
        combined_state = TensorDict({"c": c_state, "n": n_state, "m": m_state}, batch_size=[B])
        combined_state.update(conv_state)  # Add conv state
        return combined_state

    def forward(
        self,
        x: Tensor,
        state: MaybeState,
        *,
        resets: Optional[ResetMask] = None,
    ) -> Tuple[Tensor, MaybeState]:
        """Apply mLSTM with automatic backend selection for step or chunk processing."""
        # Check if single step
        is_step = x.dim() == 2
        if is_step:
            x_seq = x.unsqueeze(1)  # [B, H] -> [B, 1, H]
        else:
            x_seq = x  # [B, T, H]

        B, T, H = x_seq.shape

        # Initialize or get state
        if state is None or not all(k in state for k in ["c", "n", "m"]):
            st = self.init_state(batch=B, device=x.device, dtype=x.dtype)
        else:
            st = state

        c_state = st.get("c")  # [B, NH, DH, DH]
        n_state = st.get("n")  # [B, NH, DH, 1]
        m_state = st.get("m")  # [B, NH, 1, 1]

        # Note: mLSTM backends support reset masks directly. For step we pass a
        # [B] mask; for sequences we pass a [B, T] mask to the chunkwise backend.

        # Always run causal conv to precondition Q/K inputs
        if st is not None and "conv" in st:
            conv_state_dict = TensorDict({"conv": st.get("conv")}, batch_size=[B])
        else:
            conv_state_dict = None
        if is_step:
            x_conv, conv_state_new = self.conv1d_cell(x_seq.squeeze(1), conv_state_dict, resets=resets)
            x_conv = x_conv.unsqueeze(1)  # [B, H] -> [B, 1, H]
        else:
            x_conv, conv_state_new = self.conv1d_cell(x_seq, conv_state_dict, resets=resets)
        x_conv_act = self.conv_act(x_conv)

        # Build Q, K, V
        if not self.cfg.use_axon_qkv:
            # Q/K from conv, V is raw input
            q = x_conv_act
            k = x_conv_act
            v = x_seq
        else:
            # Axon-backed Q,K use conv-preconditioned input; V from raw input
            qk = self.qk_layer(x_conv_act, state=st, resets=resets)
            q = self.qkv_act(qk)
            k = self.qkv_act(qk)
            v = self.v_layer(x_seq, state=st, resets=resets)

        if_gate_input = torch.cat([q, k, v], dim=-1)

        # Reshape Q, K, V
        q = q.view(B, T, self.cfg.num_heads, self.head_dim)  # [B, T, NH, DH]
        k = k.view(B, T, self.cfg.num_heads, self.head_dim)  # [B, T, NH, DH]
        v = v.view(B, T, self.cfg.num_heads, self.head_dim)  # [B, T, NH, DH]

        # Transpose for processing
        q = q.transpose(1, 2)  # [B, NH, T, DH]
        k = k.transpose(1, 2)  # [B, NH, T, DH]
        v = v.transpose(1, 2)  # [B, NH, T, DH]

        # Compute gates
        if self.cfg.use_axon_layer:
            igate_preact = self.igate(if_gate_input, state=st, resets=resets)  # [B, T, NH]
            fgate_preact = self.fgate(if_gate_input, state=st, resets=resets)  # [B, T, NH]
        else:
            igate_preact = self.igate(if_gate_input)  # [B, T, NH]
            fgate_preact = self.fgate(if_gate_input)  # [B, T, NH]
        igate_preact = igate_preact.transpose(-1, -2)  # [B, NH, T]
        fgate_preact = fgate_preact.transpose(-1, -2)  # [B, NH, T]

        if is_step:
            # Single step recurrent processing
            igate_preact = igate_preact.unsqueeze(-1)  # [B, NH, T, 1]
            fgate_preact = fgate_preact.unsqueeze(-1)  # [B, NH, T, 1]

            # Prepare a step reset mask if provided
            reset_step: Optional[torch.Tensor]
            if resets is None:
                reset_step = None
            else:
                # Accept [B] or [B, 1] and convert to [B]
                reset_step = resets.view(B)

            # Step mode always uses PyTorch (no Triton step kernel)
            h_state, (c_new, n_new, m_new) = mlstm_recurrent_step_stabilized_simple(
                c_state=c_state,
                n_state=n_state,
                m_state=m_state,
                q=q,
                k=k,
                v=v,
                igate_preact=igate_preact,
                fgate_preact=fgate_preact,
                reset_mask=reset_step,
            )
            new_state = TensorDict({"c": c_new, "n": n_new, "m": m_new}, batch_size=[B])
            if conv_state_new is not None:
                new_state.update(conv_state_new)
            # Preserve any auxiliary substates (e.g., AxonLayer groups written into `st`)
            try:
                for k in list(st.keys()):
                    if k not in ("c", "n", "m"):
                        new_state[k] = st.get(k)
            except Exception:
                pass
        else:
            # Sequence processing
            # Exact segment-aware handling for resets: split the sequence into
            # independent segments at reset positions and run each segment with
            # zero initial state. This matches step semantics exactly while
            # still allowing chunkwise acceleration within each segment.
            if resets is not None and (resets.any().item() if isinstance(resets, torch.Tensor) else bool(resets)):
                # Normalize resets to (B, T)
                if resets.dim() == 1:
                    rm = torch.zeros(B, T, dtype=resets.dtype, device=x.device)
                    rm[:, 0] = resets
                else:
                    rm = resets

                backend_fn = select_backend(
                    triton_fn=mlstm_chunkwise_triton,
                    pytorch_fn=mlstm_chunkwise_simple,
                    tensor=x,
                    allow_triton=True,
                )
                h_state, (c_new, n_new, m_new) = backend_fn(
                    queries=q,
                    keys=k,
                    values=v,
                    igate_preact=igate_preact,
                    fgate_preact=fgate_preact,
                    initial_C=c_state,
                    initial_n=n_state,
                    initial_m=m_state,
                    reset_mask=rm,
                    chunk_size=self.cfg.chunk_size,
                    return_last_state=True,
                )
            else:
                # No resets: use standard chunkwise backend
                backend_kwargs = {
                    "queries": q,
                    "keys": k,
                    "values": v,
                    "igate_preact": igate_preact,
                    "fgate_preact": fgate_preact,
                    "initial_C": c_state,
                    "initial_n": n_state,
                    "initial_m": m_state,
                    "chunk_size": self.cfg.chunk_size,
                    "return_last_state": True,
                }

                backend_fn = select_backend(
                    triton_fn=mlstm_chunkwise_triton,
                    pytorch_fn=mlstm_chunkwise_simple,
                    tensor=x,
                    allow_triton=True,
                )
                h_state, (c_new, n_new, m_new) = backend_fn(**backend_kwargs)
            # Attach conv buffer after sequence for continuity across calls
            new_state = TensorDict({"c": c_new, "n": n_new, "m": m_new}, batch_size=[B])
            if conv_state_new is not None:
                new_state.update(conv_state_new)
            # Preserve any auxiliary substates (e.g., AxonLayer groups written into `st`)
            try:
                for k in list(st.keys()):
                    if k not in ("c", "n", "m"):
                        new_state[k] = st.get(k)
            except Exception:
                pass

        # Apply output normalization
        h_state_norm = self.outnorm(h_state)  # [B, NH, T, DH]
        h_state_norm = h_state_norm.transpose(1, 2).reshape(B, T, -1)  # [B, T, H]

        # Return in original shape
        if is_step:
            return h_state_norm.squeeze(1), new_state  # [B, H]
        else:
            return h_state_norm, new_state  # [B, T, H]

    def reset_state(self, state: MaybeState, mask: ResetMask) -> MaybeState:
        """Reset state for masked batch elements."""
        if state is None:
            return state

        mask_expanded = mask.to(dtype=state["c"].dtype).view(-1, 1, 1, 1)
        state["c"] = state["c"] * (1.0 - mask_expanded)
        state["n"] = state["n"] * (1.0 - mask_expanded)
        state["m"] = state["m"] * (1.0 - mask_expanded)

        # Reset conv state using the CausalConv1d cell's reset_state method (if used)
        if self.conv1d_cell is not None and "conv" in state:
            conv_state_dict = TensorDict({"conv": state["conv"]}, batch_size=[state["c"].shape[0]])
            conv_state_dict = self.conv1d_cell.reset_state(conv_state_dict, mask)
            # Avoid boolean conversion of TensorDict
            if "conv" in conv_state_dict:
                state["conv"] = conv_state_dict["conv"]

        # Reset Axon gate substates if AxonLayer is enabled
        if self.cfg.use_axon_layer:
            self.igate.reset_state(mask, state)
            self.fgate.reset_state(mask, state)

        # Reset Axon QKV substates when enabled (shared QK + V)
        if self.cfg.use_axon_qkv:
            self.qk_layer.reset_state(mask, state)
            self.v_layer.reset_state(mask, state)

        return state


__all__ = ["mLSTMCell", "mLSTMCellConfig"]
