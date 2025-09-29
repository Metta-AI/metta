from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict

from cortex.cells.backends import (
    MultiHeadLayerNorm,
    bias_linspace_init_,
    mlstm_chunkwise_simple,
    mlstm_recurrent_step_stabilized_simple,
)
from cortex.cells.base import MemoryCell
from cortex.cells.conv import CausalConv1d
from cortex.cells.registry import register_cell
from cortex.config import CausalConv1dConfig, mLSTMCellConfig
from cortex.types import MaybeState, ResetMask, Tensor


@register_cell(mLSTMCellConfig)
class mLSTMCell(MemoryCell):
    """Stateful mLSTM cell with matrix-valued state.

    The mLSTM cell maintains three state components:
    - c_state: [B, num_heads, head_dim, head_dim] - matrix state
    - n_state: [B, num_heads, head_dim, 1] - normalization vector
    - m_state: [B, num_heads, 1, 1] - max log gate value

    The cell supports both parallel processing of sequences and
    step-by-step recurrent processing with persistent state.
    """

    def __init__(self, cfg: mLSTMCellConfig) -> None:
        super().__init__(hidden_size=cfg.hidden_size)
        self.cfg = cfg

        assert cfg.hidden_size % cfg.num_heads == 0, "hidden_size must be divisible by num_heads"
        self.head_dim = cfg.hidden_size // cfg.num_heads

        # Input/forget gates - exact as in original
        self.igate = nn.Linear(3 * cfg.hidden_size, cfg.num_heads)
        self.fgate = nn.Linear(3 * cfg.hidden_size, cfg.num_heads)

        # Causal depthwise Conv1d on input, used to form Q/K.
        # Match reference pattern: q,k from conv-activated features; v from raw features.
        self.conv_kernel_size = cfg.conv1d_kernel_size
        conv_config = CausalConv1dConfig(
            hidden_size=cfg.hidden_size,
            kernel_size=self.conv_kernel_size,
            causal_conv_bias=True,
            channel_mixing=False,  # depthwise convolution
        )
        self.conv1d_cell = CausalConv1d(conv_config)
        self.conv_act = nn.SiLU()

        # Output normalization
        self.outnorm = MultiHeadLayerNorm(cfg.hidden_size, weight=True, bias=False)

        # Backend functions
        self.backend_fn = mlstm_chunkwise_simple
        self.backend_fn_step = mlstm_recurrent_step_stabilized_simple

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        self.outnorm.reset_parameters()
        self.conv1d_cell.reset_parameters()
        # Forget gate initialization
        torch.nn.init.zeros_(self.fgate.weight)
        bias_linspace_init_(self.fgate.bias, start=3.0, end=6.0)
        # Input gate initialization
        torch.nn.init.zeros_(self.igate.weight)
        torch.nn.init.normal_(self.igate.bias, mean=0.0, std=0.1)

    def init_state(self, batch: int, *, device: torch.device | str, dtype: torch.dtype) -> TensorDict:
        """Initialize state tensors."""
        B = batch
        NH = self.cfg.num_heads
        DH = self.head_dim

        c_state = torch.zeros(B, NH, DH, DH, device=device, dtype=dtype)
        n_state = torch.zeros(B, NH, DH, 1, device=device, dtype=dtype)
        m_state = torch.zeros(B, NH, 1, 1, device=device, dtype=dtype)

        # Get conv state from the conv1d cell
        conv_state = self.conv1d_cell.init_state(batch, device=device, dtype=dtype)

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
        """Forward pass with optional state and reset handling.

        Args:
            x: Input tensor [B, T, H] or [B, H]
            state: Optional state TensorDict with c, n, m tensors
            resets: Optional reset mask [B] or [B, T]

        Returns:
            Tuple of (output [B, T, H] or [B, H], new_state)
        """
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

        # Apply resets if provided
        if resets is not None:
            if is_step:
                mask = resets.to(dtype=x.dtype).view(B, 1, 1, 1)
                c_state = c_state * (1.0 - mask)
                n_state = n_state * (1.0 - mask)
                m_state = m_state * (1.0 - mask)
            else:
                # For sequences, we need to handle resets per timestep
                # TODO: This is simplified - proper implementation would reset before each timestep
                pass

        # Causal conv on input to form q/k (v remains raw input)
        # Extract conv state from the combined state
        if st is not None and "conv" in st:
            conv_state_dict = TensorDict({"conv": st.get("conv")}, batch_size=[B])
        else:
            conv_state_dict = None

        # Apply convolution using the CausalConv1d cell
        if is_step:
            x_conv, conv_state_new = self.conv1d_cell(x_seq.squeeze(1), conv_state_dict, resets=resets)
            x_conv = x_conv.unsqueeze(1)  # [B, H] -> [B, 1, H]
        else:
            x_conv, conv_state_new = self.conv1d_cell(x_seq, conv_state_dict, resets=resets)

        x_conv_act = self.conv_act(x_conv)

        # q, k from conv-activated path; v from raw input (reference pattern)
        q = x_conv_act
        k = x_conv_act
        v = x_seq

        if_gate_input = torch.cat([q, k, v], dim=-1)

        # Reshape Q, K, V
        q = q.view(B, T, self.cfg.num_heads, self.head_dim)  # [B, T, NH, DH]
        k = k.view(B, T, self.cfg.num_heads, self.head_dim)  # [B, T, NH, DH]
        v = v.view(B, T, self.cfg.num_heads, self.head_dim)  # [B, T, NH, DH]

        # Transpose for processing
        q = q.transpose(1, 2)  # [B, NH, T, DH]
        k = k.transpose(1, 2)  # [B, NH, T, DH]
        v = v.transpose(1, 2)  # [B, NH, T, DH]

        # Compute gates - exact as in original
        igate_preact = self.igate(if_gate_input)  # [B, T, NH]
        igate_preact = igate_preact.transpose(-1, -2)  # [B, NH, T]
        fgate_preact = self.fgate(if_gate_input)  # [B, T, NH]
        fgate_preact = fgate_preact.transpose(-1, -2)  # [B, NH, T]

        if is_step:
            # Single step recurrent processing
            igate_preact = igate_preact.unsqueeze(-1)  # [B, NH, T, 1]
            fgate_preact = fgate_preact.unsqueeze(-1)  # [B, NH, T, 1]

            h_state, (c_new, n_new, m_new) = self.backend_fn_step(
                c_state=c_state,
                n_state=n_state,
                m_state=m_state,
                q=q,
                k=k,
                v=v,
                igate_preact=igate_preact,
                fgate_preact=fgate_preact,
            )
            new_state = TensorDict({"c": c_new, "n": n_new, "m": m_new}, batch_size=[B])
            new_state.update(conv_state_new)  # Add updated conv state
        else:
            # Parallel processing for sequences using chunkwise backend
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

            backend_output = self.backend_fn(**backend_kwargs)

            h_state, (c_new, n_new, m_new) = backend_output
            # Attach conv buffer after sequence for continuity across calls
            new_state = TensorDict({"c": c_new, "n": n_new, "m": m_new}, batch_size=[B])
            new_state.update(conv_state_new)  # Add updated conv state

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

        # Reset conv state using the CausalConv1d cell's reset_state method
        if "conv" in state:
            conv_state_dict = TensorDict({"conv": state["conv"]}, batch_size=[state["c"].shape[0]])
            conv_state_dict = self.conv1d_cell.reset_state(conv_state_dict, mask)
            # Avoid boolean conversion of TensorDict
            if "conv" in conv_state_dict:
                state["conv"] = conv_state_dict["conv"]

        return state


__all__ = ["mLSTMCell", "mLSTMCellConfig"]
