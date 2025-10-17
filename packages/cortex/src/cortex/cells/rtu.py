"""RTU cell wrapper with clean functional kernel interfaces.

Parameters live in the cell; kernels are pure functions in
`cortex.kernels.pytorch.rtu` and `cortex.kernels.triton.rtu`.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict

from cortex.cells.base import MemoryCell
from cortex.cells.registry import register_cell
from cortex.config import RTUCellConfig
from cortex.kernels.pytorch.rtu import rtu_sequence_pytorch
from cortex.kernels.triton.rtu import rtu_sequence_triton  # type: ignore
from cortex.types import MaybeState, ResetMask, Tensor
from cortex.utils import select_backend


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
      - calls functional RTU kernels to compute 2H activations,
      - projects `2H -> H` to fit Cortex block shapes,
      - manages TensorDict state with keys {"hc1", "hc2"},
      - handles step-vs-sequence inputs and reset masks.
    """

    def __init__(self, cfg: RTUCellConfig) -> None:
        if cfg.hidden_size is None:
            raise ValueError("RTUCellConfig.hidden_size must be set")
        super().__init__(hidden_size=cfg.hidden_size)
        self.cfg = cfg

        H = cfg.hidden_size
        self.activation = _resolve_activation(cfg.activation)

        # Parameters for RTU dynamics and low-rank maps
        u1 = torch.rand(H)
        inner = u1 * (cfg.r_max**2 - cfg.r_min**2) + cfg.r_min**2
        nu_log_init = torch.log(-0.5 * torch.log(inner.clamp(min=1e-12)))
        u2 = torch.rand(H)
        theta_log_init = torch.log((cfg.max_phase * u2).clamp(min=1e-12))
        self.nu_log = nn.Parameter(nu_log_init)
        self.theta_log = nn.Parameter(theta_log_init)

        self.U1 = nn.Parameter(torch.empty(H, cfg.rank))
        self.U2 = nn.Parameter(torch.empty(H, cfg.rank))
        self.V1 = nn.Parameter(torch.empty(cfg.rank, H))
        self.V2 = nn.Parameter(torch.empty(cfg.rank, H))

        bound_in = 1.0 / math.sqrt(H)
        bound_r = 1.0 / math.sqrt(cfg.rank)
        with torch.no_grad():
            self.U1.uniform_(-bound_in, bound_in)
            self.U2.uniform_(-bound_in, bound_in)
            self.V1.uniform_(-bound_r, bound_r)
            self.V2.uniform_(-bound_r, bound_r)

        # Project 2H -> H to retain external block shape compatibility
        self.out_proj = nn.Linear(2 * H, H)

    def init_state(self, batch: int, *, device: torch.device | str, dtype: torch.dtype) -> TensorDict:
        H = self.hidden_size
        zero = torch.zeros(batch, H, device=device, dtype=dtype)
        return TensorDict({"hc1": zero.clone(), "hc2": zero.clone()}, batch_size=[batch])

    def forward(
        self,
        x: Tensor,
        state: MaybeState,
        *,
        resets: Optional[ResetMask] = None,
    ) -> Tuple[Tensor, MaybeState]:
        # Determine step vs sequence and normalize inputs
        is_step = x.dim() == 2  # [B,H] vs [B,T,H]
        if is_step:
            x_btd = x.unsqueeze(1)
        else:
            x_btd = x
        B, T, H = x_btd.shape

        # Prepare/validate state
        if state is None or not all(k in state for k in ("hc1", "hc2")):
            st = self.init_state(batch=B, device=x_btd.device, dtype=x_btd.dtype)
        else:
            st = state

        hc1 = st.get("hc1")
        hc2 = st.get("hc2")
        assert hc1 is not None and hc2 is not None

        # Normalize resets mask
        resets_bt: Optional[torch.Tensor]
        if resets is None:
            resets_bt = None
        else:
            if is_step:
                resets_bt = resets.view(B, 1)
            else:
                resets_bt = resets if resets.dim() == 2 else resets.view(B, 1).expand(B, T)

        # Backend shims
        def _fw_triton(x_in: torch.Tensor, h1: torch.Tensor, h2: torch.Tensor, rm: Optional[torch.Tensor]):
            act_name = self.activation.__class__.__name__
            y2h_t, (h1_n, h2_n) = rtu_sequence_triton(
                x_btd=x_in,
                nu_log=self.nu_log,
                theta_log=self.theta_log,
                U1=self.U1,
                U2=self.U2,
                V1=self.V1,
                V2=self.V2,
                activation_name=act_name,
                hc1_init_bh=h1,
                hc2_init_bh=h2,
                resets_bt=rm,
            )
            return y2h_t, h1_n, h2_n

        def _fw_torch(x_in: torch.Tensor, h1: torch.Tensor, h2: torch.Tensor, rm: Optional[torch.Tensor]):
            act_name = self.activation.__class__.__name__
            y2h_t, (h1_n, h2_n) = rtu_sequence_pytorch(
                x_btd=x_in,
                nu_log=self.nu_log,
                theta_log=self.theta_log,
                U1=self.U1,
                U2=self.U2,
                V1=self.V1,
                V2=self.V2,
                activation_name=act_name,
                hc1_init_bh=h1,
                hc2_init_bh=h2,
                resets_bt=rm,
            )
            return y2h_t, h1_n, h2_n

        backend = select_backend(
            triton_fn=_fw_triton,
            pytorch_fn=_fw_torch,
            tensor=x_btd,
            allow_triton=True,
        )

        y2h, hc1_n, hc2_n = backend(x_btd, hc1, hc2, resets_bt)
        # Project 2H -> H (batch-first)
        if is_step:
            y = self.out_proj(y2h.squeeze(1))
        else:
            y = self.out_proj(y2h.reshape(B * T, -1)).reshape(B, T, self.hidden_size)

        return y, TensorDict({"hc1": hc1_n, "hc2": hc2_n}, batch_size=[B])

    def reset_state(self, state: MaybeState, mask: ResetMask) -> MaybeState:
        if state is None:
            return None
        # Broadcast to [B, 1]
        m = mask.to(dtype=state["hc1"].dtype).view(-1, 1)
        state["hc1"] = state["hc1"] * (1.0 - m)
        state["hc2"] = state["hc2"] * (1.0 - m)
        return state


__all__ = ["RTUCell"]
