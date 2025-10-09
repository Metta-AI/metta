"""RTU cell wrapper integrating the PyTorch low-rank RTU kernel with Cortex APIs."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict

from cortex.cells.base import MemoryCell
from cortex.cells.registry import register_cell
from cortex.config import RTUCellConfig
from cortex.kernels.pytorch.rtu import LinearRTU
from cortex.utils import select_backend

# Import Triton autograd Function directly to reuse the same Parameters
try:
    from cortex.kernels.triton.rtu import _LinearRTUFunctionLR_Triton  # type: ignore
except Exception:  # pragma: no cover - Triton optional
    _LinearRTUFunctionLR_Triton = None  # type: ignore[assignment]
from cortex.types import MaybeState, ResetMask, Tensor


def _resolve_activation(name: str) -> nn.Module:
    n = name.lower()
    if n in ("silu", "swish"):
        return nn.SiLU()
    if n == "relu":
        return nn.ReLU()
    if n == "tanh":
        return nn.Tanh()
    if n in ("linear", "identity"):
        return nn.Identity()
    raise ValueError(f"Unsupported RTU activation: {name}")


@register_cell(RTUCellConfig)
class RTUCell(MemoryCell):
    """Cortex memory cell for the low-rank Recurrent Trace Unit (RTU).

    This wrapper:
      - instantiates the PyTorch kernel (`LinearRTU`) to compute the 2H activations,
      - adds a learned projection `2H -> H` so the cell fits Cortex block shapes,
      - manages TensorDict state with keys {"hc1", "hc2"},
      - handles step-vs-sequence inputs and reset masks.
    """

    def __init__(self, cfg: RTUCellConfig) -> None:
        if cfg.hidden_size is None:
            raise ValueError("RTUCellConfig.hidden_size must be set")
        super().__init__(hidden_size=cfg.hidden_size)
        self.cfg = cfg

        H = cfg.hidden_size
        act = _resolve_activation(cfg.activation)
        # Kernel runs batch-first inside the cell (PyTorch baseline)
        self.core = LinearRTU(
            input_size=H,
            hidden_size=H,
            rank=cfg.rank,
            batch_first=True,
            activation=act,
            r_max=cfg.r_max,
            r_min=cfg.r_min,
            max_phase=cfg.max_phase,
        )

        # Project 2H -> H to retain external block shape compatibility
        self.out_proj = nn.Linear(2 * H, H)

    def init_state(self, batch: int, *, device: torch.device | str, dtype: torch.dtype) -> TensorDict:
        H = self.hidden_size
        zero = torch.zeros(batch, H, device=device, dtype=dtype)
        return TensorDict({"hc1": zero.clone(), "hc2": zero.clone()}, batch_size=[batch])

    def _apply_core_sequence(
        self, x_seq: torch.Tensor, hc1: torch.Tensor, hc2: torch.Tensor, *, resets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x_seq: [B, T, H]
        # Define backends: Triton function uses same Parameters from PyTorch module
        def _fw_triton(x: torch.Tensor, h1: torch.Tensor, h2: torch.Tensor, rm: Optional[torch.Tensor]):
            assert _LinearRTUFunctionLR_Triton is not None, "Triton backend not available"
            act_name = self.core.activation.__class__.__name__
            y2h_t, h1_n, h2_n = _LinearRTUFunctionLR_Triton.apply(
                x,
                self.core.nu_log,
                self.core.theta_log,
                self.core.U1,
                self.core.U2,
                self.core.V1,
                self.core.V2,
                act_name,
                h1,
                h2,
                rm,
                1,  # param_parallel: use fully-parallel path by default
            )
            return y2h_t, h1_n, h2_n

        def _fw_torch(x: torch.Tensor, h1: torch.Tensor, h2: torch.Tensor, rm: Optional[torch.Tensor]):
            return self.core(x, (h1, h2), resets=rm)

        backend = select_backend(
            triton_fn=_fw_triton if _LinearRTUFunctionLR_Triton is not None else None,
            pytorch_fn=_fw_torch,
            tensor=x_seq,
            allow_triton=True,
        )

        y2h, hc1_n, hc2_n = backend(x_seq, hc1, hc2, resets)
        # Map 2H -> H (batch-first)
        B, T, _ = y2h.shape
        y = self.out_proj(y2h.reshape(B * T, -1)).reshape(B, T, self.hidden_size)
        return y, hc1_n, hc2_n

    def _apply_core_step(
        self, x_step: torch.Tensor, hc1: torch.Tensor, hc2: torch.Tensor, *, resets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x_step: [B, H]
        # Normalize resets to [B, 1] if provided
        resets_bt = None
        if resets is not None:
            resets_bt = resets.view(-1, 1)
        # Choose backend as in sequence path
        def _fw_triton(x: torch.Tensor, h1: torch.Tensor, h2: torch.Tensor, rm: Optional[torch.Tensor]):
            assert _LinearRTUFunctionLR_Triton is not None, "Triton backend not available"
            act_name = self.core.activation.__class__.__name__
            y2h_t, h1_n, h2_n = _LinearRTUFunctionLR_Triton.apply(
                x,
                self.core.nu_log,
                self.core.theta_log,
                self.core.U1,
                self.core.U2,
                self.core.V1,
                self.core.V2,
                act_name,
                h1,
                h2,
                rm,
                1,
            )
            return y2h_t, h1_n, h2_n

        def _fw_torch(x: torch.Tensor, h1: torch.Tensor, h2: torch.Tensor, rm: Optional[torch.Tensor]):
            return self.core(x, (h1, h2), resets=rm)

        backend = select_backend(
            triton_fn=_fw_triton if _LinearRTUFunctionLR_Triton is not None else None,
            pytorch_fn=_fw_torch,
            tensor=x_step,
            allow_triton=True,
        )

        y2h, hc1_n, hc2_n = backend(x_step.unsqueeze(1), hc1, hc2, resets_bt)
        y = self.out_proj(y2h.squeeze(1))
        return y, hc1_n, hc2_n

    def forward(
        self,
        x: Tensor,
        state: MaybeState,
        *,
        resets: Optional[ResetMask] = None,
    ) -> Tuple[Tensor, MaybeState]:
        # Determine step vs sequence
        is_step = x.dim() == 2
        if is_step:
            x_seq = x
            B = x_seq.shape[0]
        else:
            x_seq = x  # [B, T, H]
            B, T, _ = x_seq.shape

        # Prepare/validate state
        if state is None or not all(k in state for k in ("hc1", "hc2")):
            st = self.init_state(batch=x_seq.shape[0], device=x_seq.device, dtype=x_seq.dtype)
        else:
            st = state

        hc1 = st.get("hc1")
        hc2 = st.get("hc2")
        assert hc1 is not None and hc2 is not None

        # Unified path: delegate resets handling to the kernel
        if is_step:
            y, hc1_n, hc2_n = self._apply_core_step(x_seq, hc1, hc2, resets=resets)
            return y, TensorDict({"hc1": hc1_n, "hc2": hc2_n}, batch_size=[x_seq.shape[0]])
        else:
            # Normalize resets to [B, T] if provided
            resets_bt = None
            if resets is not None:
                resets_bt = resets if resets.dim() == 2 else resets.view(-1, 1).expand(x_seq.shape[0], x_seq.shape[1])
            y, hc1_n, hc2_n = self._apply_core_sequence(x_seq, hc1, hc2, resets=resets_bt)
            return y, TensorDict({"hc1": hc1_n, "hc2": hc2_n}, batch_size=[x_seq.shape[0]])

    def reset_state(self, state: MaybeState, mask: ResetMask) -> MaybeState:
        if state is None:
            return None
        # Broadcast to [B, 1]
        m = mask.to(dtype=state["hc1"].dtype).view(-1, 1)
        state["hc1"] = state["hc1"] * (1.0 - m)
        state["hc2"] = state["hc2"] * (1.0 - m)
        return state


__all__ = ["RTUCell"]
