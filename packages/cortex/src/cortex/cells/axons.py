"""Axons cell: streaming RTU (diagonal input weights, PyTorch/Triton/CUDA).

This cell mirrors the structure of `cortex.cells.rtu.RTUCell` and can run on
either the PyTorch or Triton backend. For streaming diagonal RTU it uses:

- PyTorch: `cortex.kernels.pytorch.rtu_stream.rtu_stream_diag_pytorch`
- Triton:  `cortex.kernels.triton.rtu.stream_diag.rtu_stream_diag_triton` (CUDA)
- CUDA (seq-allin, short-T): `cortex.kernels.cuda.rtu_stream_diag_cuda_seq_allin` (CUDA)

Notes
-----
- Assumes D == H (identity input map) as enforced by the kernel.
- Carries compact [B,H] eligibility traces across chunks. The traces are
  included in the returned TensorDict state and can be detached between
  subsequences by the caller to achieve true streaming.
- Backend selection follows `cortex.utils.select_backend`: Triton is used when
  available on CUDA unless disabled; otherwise falls back to PyTorch.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict

from cortex.backends import load_cuda_stream_diag, want_cuda_seq_allin
from cortex.cells.base import MemoryCell
from cortex.cells.registry import register_cell
from cortex.config import AxonsConfig
from cortex.kernels.cuda.srht_cuda import srht_cuda  # type: ignore
from cortex.kernels.pytorch.rtu_stream import rtu_stream_diag_pytorch
from cortex.kernels.pytorch.srht import srht_pytorch
from cortex.kernels.triton.rtu import rtu_stream_diag_triton  # type: ignore
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


@register_cell(AxonsConfig)
class Axons(MemoryCell):
    """Cortex memory cell for streaming RTU with diagonal input weights.

    This wrapper:
      - calls the streaming RTU PyTorch kernel (diagonal w1/w2 input weights),
      - projects `2H -> H` to fit Cortex block shapes,
      - manages TensorDict state with keys {"hc1", "hc2", trace tensors},
      - handles step-vs-sequence inputs and reset masks.
    """

    def __init__(self, cfg: AxonsConfig) -> None:
        if cfg.hidden_size is None:
            raise ValueError("AxonsConfig.hidden_size must be set")
        super().__init__(hidden_size=cfg.hidden_size)
        self.cfg = cfg

        H = cfg.hidden_size
        self.activation = _resolve_activation(cfg.activation)

        # Dynamics parameters (exp-exp parameterization)
        u1 = torch.rand(H)
        inner = u1 * (cfg.r_max**2 - cfg.r_min**2) + cfg.r_min**2
        nu_log_init = torch.log(-0.5 * torch.log(inner.clamp(min=1e-12)))
        u2 = torch.rand(H)
        theta_log_init = torch.log((cfg.max_phase * u2).clamp(min=1e-12))
        self.nu_log = nn.Parameter(nu_log_init)
        self.theta_log = nn.Parameter(theta_log_init)

        # Diagonal input weights (per-channel)
        self.w1 = nn.Parameter(torch.empty(H))
        self.w2 = nn.Parameter(torch.empty(H))
        bound_in = 1.0 / math.sqrt(H)
        with torch.no_grad():
            self.w1.uniform_(-bound_in, bound_in)
            self.w2.uniform_(-bound_in, bound_in)

        # Low-rank output projection: 2H -> r -> out_dim (defaults to H).
        out_dim = cfg.out_dim if getattr(cfg, "out_dim", None) not in (None, 0) else H
        # Rank rule: if specified, use it; otherwise use maximum possible rank for (2H -> out_dim)
        if getattr(cfg, "out_rank", None) is None:
            r = min(2 * H, out_dim)
        else:
            r = int(cfg.out_rank)
        if r < 1:
            raise ValueError(f"Axons out_rank must be >= 1, got {r}")
        self._out_dim = out_dim
        self._out_rank = r
        self.out_lr1 = nn.Linear(2 * H, r, bias=False)
        self.out_lr2 = nn.Linear(r, out_dim, bias=True)

        # SRHT mixer parameters (fixed buffers)
        self._use_srht = bool(getattr(cfg, "use_srht", False))
        if self._use_srht:
            rng = torch.Generator(device="cpu")
            rng.manual_seed(0)
            signs = torch.empty(H, dtype=torch.float32).bernoulli_(0.5, generator=rng) * 2 - 1
            self.register_buffer("srht_signs", signs)
            if bool(getattr(cfg, "srht_permute", True)):
                perm = torch.randperm(H, generator=rng, dtype=torch.int64)
                self.register_buffer("srht_perm", perm)
            else:
                self.register_buffer("srht_perm", torch.arange(H, dtype=torch.int64))
        else:
            self.register_buffer("srht_signs", torch.empty(0))
            self.register_buffer("srht_perm", torch.empty(0, dtype=torch.int64))

    def _zero_traces(self, batch: int, *, device: torch.device | str, dtype: torch.dtype) -> TensorDict:
        H = self.hidden_size
        zero = torch.zeros(batch, H, device=device, dtype=dtype)
        return TensorDict(
            {
                "E_nu_c1": zero.clone(),
                "E_nu_c2": zero.clone(),
                "E_th_c1": zero.clone(),
                "E_th_c2": zero.clone(),
                "E_w1_c1": zero.clone(),
                "E_w1_c2": zero.clone(),
                "E_w2_c1": zero.clone(),
                "E_w2_c2": zero.clone(),
            },
            batch_size=[batch],
        )

    def init_state(self, batch: int, *, device: torch.device | str, dtype: torch.dtype) -> TensorDict:
        H = self.hidden_size
        zero = torch.zeros(batch, H, device=device, dtype=dtype)
        traces = self._zero_traces(batch, device=device, dtype=dtype)
        state = TensorDict({"hc1": zero.clone(), "hc2": zero.clone()}, batch_size=[batch])
        state.update(traces)
        return state

    def _pack_trace_in(self, state: TensorDict) -> tuple[torch.Tensor, ...] | None:
        keys = [
            "E_nu_c1",
            "E_nu_c2",
            "E_th_c1",
            "E_th_c2",
            "E_w1_c1",
            "E_w1_c2",
            "E_w2_c1",
            "E_w2_c2",
        ]
        if not all(k in state for k in keys):
            return None
        return tuple(state[k] for k in keys)  # type: ignore[return-value]

    def _unpack_trace_out(self, state: TensorDict, trace_out: tuple[torch.Tensor, ...]) -> None:
        (
            E_nu_c1,
            E_nu_c2,
            E_th_c1,
            E_th_c2,
            E_w1_c1,
            E_w1_c2,
            E_w2_c1,
            E_w2_c2,
        ) = trace_out
        state["E_nu_c1"] = E_nu_c1
        state["E_nu_c2"] = E_nu_c2
        state["E_th_c1"] = E_th_c1
        state["E_th_c2"] = E_th_c2
        state["E_w1_c1"] = E_w1_c1
        state["E_w1_c2"] = E_w1_c2
        state["E_w2_c1"] = E_w2_c1
        state["E_w2_c2"] = E_w2_c2

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

        # Pack carried traces (if present)
        trace_in = self._pack_trace_in(st)

        # Optional SRHT mixer before kernel
        if getattr(self, "_use_srht", False):
            Hh = self.hidden_size
            perm = None if self.srht_perm.numel() == 0 else self.srht_perm.to(device=x_btd.device)
            signs = self.srht_signs.to(device=x_btd.device, dtype=x_btd.dtype)
            if x_btd.is_cuda and (Hh & (Hh - 1)) == 0:
                x_btd = srht_cuda(x_btd, signs, perm, normalize=True)
            else:
                x_btd = srht_pytorch(x_btd, signs, perm, normalize=True)

        # Backend shims: all return (y2h_t, h1_n, h2_n, trace_out)
        def _fw_triton(
            x_in: torch.Tensor, h1: torch.Tensor, h2: torch.Tensor, rm: Optional[torch.Tensor], tr: Optional[tuple]
        ):
            act_name = self.activation.__class__.__name__
            y2h_t_, (h1_n_, h2_n_), trace_out_ = rtu_stream_diag_triton(
                x_btd=x_in,
                nu_log=self.nu_log,
                theta_log=self.theta_log,
                w1=self.w1,
                w2=self.w2,
                activation_name=act_name,
                hc1_init_bh=h1,
                hc2_init_bh=h2,
                trace_in=tr,
                resets_bt=rm,
            )
            return y2h_t_, h1_n_, h2_n_, trace_out_

        def _fw_torch(
            x_in: torch.Tensor, h1: torch.Tensor, h2: torch.Tensor, rm: Optional[torch.Tensor], tr: Optional[tuple]
        ):
            act_name = self.activation.__class__.__name__
            y2h_t_, (h1_n_, h2_n_), trace_out_ = rtu_stream_diag_pytorch(
                x_btd=x_in,
                nu_log=self.nu_log,
                theta_log=self.theta_log,
                w1=self.w1,
                w2=self.w2,
                activation_name=act_name,
                hc1_init_bh=h1,
                hc2_init_bh=h2,
                trace_in=tr,
                resets_bt=rm,
            )
            return y2h_t_, h1_n_, h2_n_, trace_out_

        def _fw_cuda(
            x_in: torch.Tensor, h1: torch.Tensor, h2: torch.Tensor, rm: Optional[torch.Tensor], tr: Optional[tuple]
        ):
            act_name = self.activation.__class__.__name__
            cu_fn = load_cuda_stream_diag()
            assert cu_fn is not None, "CUDA stream-diag kernel not available"
            y2h_t_, (h1_n_, h2_n_), trace_out_ = cu_fn(
                x_btd=x_in,
                nu_log=self.nu_log,
                theta_log=self.theta_log,
                w1=self.w1,
                w2=self.w2,
                activation_name=act_name,
                hc1_init_bh=h1,
                hc2_init_bh=h2,
                trace_in=tr,
                resets_bt=rm,
            )
            return y2h_t_, h1_n_, h2_n_, trace_out_

        # Prefer CUDA for short sequences (<= threshold), else Triton (if available), else PyTorch.
        prefer_cuda = want_cuda_seq_allin(tensor=x_btd, seq_len=T, threshold=int(self.cfg.cuda_seq_threshold))
        backend = select_backend(
            triton_fn=_fw_triton,
            pytorch_fn=_fw_torch,
            tensor=x_btd,
            allow_triton=True,
            cuda_fn=_fw_cuda,
            allow_cuda=prefer_cuda,
        )

        y2h_t, h1_n, h2_n, trace_out = backend(x_btd, hc1, hc2, resets_bt, trace_in)

        # Update state and traces
        st["hc1"] = h1_n
        st["hc2"] = h2_n
        self._unpack_trace_out(st, trace_out)

        # Project 2H -> H (batch-first)
        if is_step:
            y2h = y2h_t.squeeze(1)
            y = self.out_lr2(self.out_lr1(y2h))
        else:
            y2h_flat = y2h_t.reshape(B * T, -1)
            y = self.out_lr2(self.out_lr1(y2h_flat)).reshape(B, T, self._out_dim)

        # For compatibility with blocks, enforce output feature = hidden_size.
        # If a user overrides out_dim != H, raise a helpful error.
        if y.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Axons out_dim={self._out_dim} != hidden_size={self.hidden_size}. "
                "Blocks expect the cell to return tensors with feature dim == hidden_size. "
                "Set AxonsConfig.out_dim to None (default) or to hidden_size."
            )

        return y, st

    def reset_state(self, state: MaybeState, mask: ResetMask) -> MaybeState:
        if state is None:
            return None
        # Broadcast to [B, 1]
        m = mask.to(dtype=state["hc1"].dtype).view(-1, 1)
        for k in list(state.keys()):
            if state[k] is None:
                continue
            if state[k].dim() == 2 and state[k].shape[0] == m.shape[0]:
                state[k] = state[k] * (1.0 - m)
        return state


__all__ = ["Axons"]
