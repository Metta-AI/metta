from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from packages.cortex.benchmarks.common import (
    BenchmarkCase,
    BenchmarkDefinition,
    BenchmarkSettings,
    measure_callable,
    register,
)

# Kernel-level streaming diagonal RTU benchmark (D == H), comparing PyTorch vs Triton
CONFIGS: Tuple[Tuple[int, int, int, bool, float], ...] = (
    (4, 128, 64, False, 0.0),
    (4, 256, 64, False, 0.0),
    (8, 256, 64, False, 0.0),
    (8, 512, 64, False, 0.0),
    (8, 512, 128, False, 0.0),
    (8, 512, 64, True, 0.1),
    (8, 1024, 64, True, 0.1),
)


def _format_config(config: Tuple[int, int, int, bool, float]) -> str:
    b, t, h, use_resets, prob = config
    return f"({b}, {t}, {h}, {use_resets}, {prob})"


def _run_case(case: BenchmarkCase, settings: BenchmarkSettings) -> Dict[str, object]:
    B, T, H, use_resets, p = case.values
    device = torch.device(settings.device)
    dtype = settings.dtype

    # Lazy import to avoid optional dependency errors when listing
    from cortex.kernels.pytorch.rtu.rtu_stream_diag import rtu_stream_diag_pytorch

    try:
        from cortex.kernels.triton.rtu import rtu_stream_diag_triton  # type: ignore
    except Exception:  # pragma: no cover - optional
        rtu_stream_diag_triton = None  # type: ignore

    D = H
    x = torch.randn(B, T, D, device=device, dtype=dtype)
    nu_log = torch.randn(H, device=device, dtype=dtype)
    theta_log = torch.randn(H, device=device, dtype=dtype)
    w1 = torch.randn(H, device=device, dtype=dtype)
    w2 = torch.randn(H, device=device, dtype=dtype)
    hc1 = torch.zeros(B, H, device=device, dtype=dtype)
    hc2 = torch.zeros(B, H, device=device, dtype=dtype)
    trace_in = None
    resets: Optional[torch.Tensor] = None
    if use_resets:
        resets = (torch.rand(B, T, device=device) < float(p)).to(device=device)

    synchronize = device.type == "cuda"

    def run_pytorch():
        y, _, _ = rtu_stream_diag_pytorch(
            x_btd=x,
            nu_log=nu_log,
            theta_log=theta_log,
            w1=w1,
            w2=w2,
            activation_name="SiLU",
            hc1_init_bh=hc1,
            hc2_init_bh=hc2,
            trace_in=trace_in,
            resets_bt=resets,
        )
        return y

    y_pt, pt_time = measure_callable(
        run_pytorch,
        warmup=settings.warmup,
        iterations=settings.iterations,
        synchronize=synchronize,
    )

    results: Dict[str, object] = {
        "pytorch_ms": pt_time * 1000.0,
        "triton_ms": None,
        "speedup": None,
        "max_diff": None,
    }

    if device.type != "cuda" or rtu_stream_diag_triton is None:
        return results

    def run_triton():
        y, _, _ = rtu_stream_diag_triton(
            x_btd=x,
            nu_log=nu_log,
            theta_log=theta_log,
            w1=w1,
            w2=w2,
            activation_name="SiLU",
            hc1_init_bh=hc1,
            hc2_init_bh=hc2,
            trace_in=trace_in,
            resets_bt=resets,
        )
        return y

    try:
        y_tr, tr_time = measure_callable(
            run_triton,
            warmup=settings.warmup,
            iterations=settings.iterations,
            synchronize=True,
        )
        results["triton_ms"] = tr_time * 1000.0
        if tr_time > 0:
            results["speedup"] = pt_time / tr_time
        results["max_diff"] = torch.max(torch.abs(y_pt - y_tr)).item()
    except Exception as exc:  # pragma: no cover - defensive
        results["error"] = str(exc)

    return results


register(
    BenchmarkDefinition(
        key="rtu",
        title="RTU Streaming Diagonal (D==H) Triton vs PyTorch",
        description=(
            "Benchmark kernel-level streaming RTU with diagonal input weights (D==H), comparing PyTorch vs Triton. "
            "Resets enable segmented-scan path."
        ),
        configs=CONFIGS,
        format_config=_format_config,
        run_case=_run_case,
        notes="CUDA-only for Triton path; CPU runs PyTorch reference only.",
    )
)
