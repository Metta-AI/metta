from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn
from cortex.cells.core import AxonCell
from cortex.config import AxonsConfig

from .common import (
    BenchmarkCase,
    BenchmarkDefinition,
    BenchmarkSettings,
    ColumnSpec,
    register,
)


@dataclass
class _BenchResult:
    ms_per_iter: float
    toks_per_s: float


def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _force_axon_backend(which: str) -> Iterator[None]:  # type: ignore[override]
    """Force a specific backend by monkeypatching the selector.

    AxonCell imports the selector into its module namespace, so we patch both
    ``cortex.utils.select_backend`` and the alias bound in
    ``cortex.cells.core.axon_cell``.
    """
    import contextlib

    import cortex.cells.core.axon_cell as cell_mod
    import cortex.utils as utils_mod

    if which == "auto":
        # no-op context manager
        @contextlib.contextmanager
        def _noop() -> Iterator[None]:
            yield

        return _noop()

    @contextlib.contextmanager
    def _ctx() -> Iterator[None]:
        orig_utils = utils_mod.select_backend
        orig_cell = getattr(cell_mod, "select_backend", None)

        def chooser(*, triton_fn=None, pytorch_fn=None, tensor=None, allow_triton=True, cuda_fn=None, allow_cuda=False):  # type: ignore[override]
            if which == "pytorch":
                return pytorch_fn
            if which == "triton":
                return triton_fn
            if which == "cuda":
                return cuda_fn
            return orig_utils(
                triton_fn=triton_fn,
                pytorch_fn=pytorch_fn,
                tensor=tensor,
                allow_triton=allow_triton,
                cuda_fn=cuda_fn,
                allow_cuda=allow_cuda,
            )

        try:
            utils_mod.select_backend = chooser  # type: ignore[assignment]
            if orig_cell is not None:
                cell_mod.select_backend = chooser  # type: ignore[assignment]
            yield
        finally:
            utils_mod.select_backend = orig_utils  # type: ignore[assignment]
            if orig_cell is not None:
                cell_mod.select_backend = orig_cell  # type: ignore[assignment]

    return _ctx()


def _benchmark_step(
    fn,
    *,
    x: torch.Tensor,
    target: torch.Tensor,
    params: Tuple[nn.Parameter, ...],
    warmup: int,
    iterations: int,
    device: torch.device,
) -> _BenchResult:
    import time

    # Warmup
    for _ in range(max(0, warmup)):
        y = fn(x)
        loss = torch.nn.functional.mse_loss(y, target)
        loss.backward()
        for p in params:
            if p.grad is not None:
                p.grad = None

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(max(1, iterations)):
        y = fn(x)
        loss = torch.nn.functional.mse_loss(y, target)
        loss.backward()
        for p in params:
            if p.grad is not None:
                p.grad = None
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / max(1, iterations)

    b, t, _ = x.shape
    return _BenchResult(ms_per_iter=dt * 1000.0, toks_per_s=(b * t) / dt)


# Config tuple: (B, T, H, with_resets, p_reset, activation, axon_backend, use_srht)
CONFIGS: Tuple[Tuple[int, int, int, bool, float, str, str, bool], ...] = (
    (8, 512, 512, False, 0.0, "identity", "auto", True),
    (8, 512, 512, True, 0.1, "identity", "auto", True),
    (8, 256, 256, False, 0.0, "identity", "auto", True),
    (8, 256, 256, False, 0.0, "identity", "pytorch", True),
    (8, 256, 256, False, 0.0, "identity", "triton", True),
    # Explicit CUDA-forced config to validate CUDA kernels end-to-end
    (8, 256, 256, False, 0.0, "identity", "cuda", True),
)


def _format_config(cfg: Tuple[int, int, int, bool, float, str, str, bool]) -> str:
    b, t, h, use_resets, p, act, backend, use_srht = cfg
    return f"(B={b}, T={t}, H={h}, resets={use_resets}, p={p}, act={act}, backend={backend}, srht={use_srht})"


def _run_case(case: BenchmarkCase, settings: BenchmarkSettings) -> Dict[str, object]:
    bsz, seqlen, hidden, with_resets, reset_prob, activation, backend, use_srht = case.values
    device = torch.device(settings.device)
    dtype = settings.dtype

    x = torch.randn(bsz, seqlen, hidden, device=device, dtype=dtype)
    target = torch.randn_like(x)
    resets = None
    if with_resets:
        resets = (torch.rand(bsz, seqlen, device=device) < float(reset_prob)).to(device=device)

    # Linear baseline
    linear = nn.Linear(hidden, hidden, bias=True).to(device=device, dtype=dtype)

    def run_linear(t: torch.Tensor) -> torch.Tensor:
        bt = t.shape[0] * t.shape[1]
        y = linear(t.reshape(bt, hidden))
        return y.reshape(t.shape[0], t.shape[1], hidden)

    # AxonCell configured as linear-like
    ax_cfg = AxonsConfig(
        hidden_size=hidden,
        activation=activation,
        cuda_seq_threshold=1000,
        use_srht=bool(use_srht),
        out_dim=hidden,
    )
    axon = AxonCell(ax_cfg).to(device=device, dtype=dtype)

    def run_axon(t: torch.Tensor) -> torch.Tensor:
        y, _ = axon(t, state=None, resets=resets)
        return y

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with _force_axon_backend(str(backend)):
        lin_res = _benchmark_step(
            run_linear,
            x=x,
            target=target,
            params=tuple(linear.parameters()),
            warmup=settings.warmup,
            iterations=settings.iterations,
            device=device,
        )
        ax_res = _benchmark_step(
            run_axon,
            x=x,
            target=target,
            params=tuple(axon.parameters()),
            warmup=settings.warmup,
            iterations=settings.iterations,
            device=device,
        )

    peak_gb: Optional[float]
    if device.type == "cuda":
        peak_gb = float(torch.cuda.max_memory_allocated()) / (1024**3)
    else:
        peak_gb = None

    return {
        "linear_ms": lin_res.ms_per_iter,
        "axon_ms": ax_res.ms_per_iter,
        "speedup": (lin_res.ms_per_iter / ax_res.ms_per_iter) if ax_res.ms_per_iter > 0 else None,
        "linear_toks_s": lin_res.toks_per_s,
        "axon_toks_s": ax_res.toks_per_s,
        "linear_params": _count_params(linear),
        "axon_params": _count_params(axon),
        "peak_mem_gib": peak_gb,
    }


register(
    BenchmarkDefinition(
        key="linear_vs_axon",
        title="PyTorch Linear vs AxonCell (forward+backward)",
        description=(
            "Compare nn.Linear(H,H) against AxonCell configured with out_dim=H on identical inputs. "
            "Reports per-iteration time, tokens/s, parameter counts, and peak CUDA memory."
        ),
        configs=CONFIGS,
        format_config=_format_config,
        run_case=_run_case,
        columns=(
            ColumnSpec("linear_ms", "Linear (ms)", lambda v: f"{float(v):.3f}"),
            ColumnSpec("axon_ms", "Axon (ms)", lambda v: f"{float(v):.3f}"),
            ColumnSpec("speedup", "Speedup", lambda v: f"{float(v):.2f}x"),
            ColumnSpec("linear_toks_s", "Linear tok/s", lambda v: f"{float(v):.1f}"),
            ColumnSpec("axon_toks_s", "Axon tok/s", lambda v: f"{float(v):.1f}"),
            ColumnSpec("linear_params", "Linear Params", lambda v: f"{int(v):,}"),
            ColumnSpec("axon_params", "Axon Params", lambda v: f"{int(v):,}"),
            ColumnSpec("peak_mem_gib", "Peak CUDA (GiB)", lambda v: f"{float(v):.2f}"),
        ),
        default_warmup=5,
        default_iterations=20,
        notes=(
            "Backend can be forced via config entries (auto|pytorch|triton|cuda). "
            "Use CUDA device for meaningful Triton/CUDA backends; CPU runs only Linear/Axon with PyTorch."
        ),
    )
)
