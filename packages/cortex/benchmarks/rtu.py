from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from packages.cortex.benchmarks.common import (
    BenchmarkCase,
    BenchmarkDefinition,
    BenchmarkSettings,
    ColumnSpec,
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

    # Optional CUDA kernel (seq-all-in variant). Use helper that returns None when unavailable.
    try:
        from cortex.backends import load_cuda_stream_diag

        rtu_stream_diag_cuda = load_cuda_stream_diag()
    except Exception:  # pragma: no cover - optional
        rtu_stream_diag_cuda = None

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
        "cuda_ms": None,
        "speedup": None,
        "speedup_cuda": None,
        "max_diff": None,
        "max_diff_cuda": None,
    }

    if device.type != "cuda":
        return results

    # Triton path if available
    if rtu_stream_diag_triton is not None:

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

    # CUDA path if available
    if rtu_stream_diag_cuda is not None:

        def run_cuda():
            y, _, _ = rtu_stream_diag_cuda(
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
            y_cu, cu_time = measure_callable(
                run_cuda,
                warmup=settings.warmup,
                iterations=settings.iterations,
                synchronize=True,
            )
            results["cuda_ms"] = cu_time * 1000.0
            if cu_time > 0:
                results["speedup_cuda"] = pt_time / cu_time
            results["max_diff_cuda"] = torch.max(torch.abs(y_pt - y_cu)).item()
        except Exception as exc:  # pragma: no cover - defensive
            results["error_cuda"] = str(exc)

    return results


register(
    BenchmarkDefinition(
        key="rtu",
        title="RTU Streaming Diagonal (D==H) PT vs Triton vs CUDA",
        description=(
            "Benchmark kernel-level streaming RTU (diagonal input weights, D==H) across PyTorch, Triton, and CUDA. "
            "Resets enable segmented-scan path."
        ),
        configs=CONFIGS,
        format_config=_format_config,
        run_case=_run_case,
        notes=(
            "Triton and CUDA paths require a CUDA device. CUDA kernel availability depends on local build; "
            "missing kernels are skipped."
        ),
        columns=(
            ColumnSpec("pytorch_ms", "PyTorch (ms)", lambda v: f"{float(v):.3f}"),
            ColumnSpec("triton_ms", "Triton (ms)", lambda v: f"{float(v):.3f}"),
            ColumnSpec("cuda_ms", "CUDA (ms)", lambda v: f"{float(v):.3f}"),
            ColumnSpec("speedup", "TRT/PT", lambda v: f"{float(v):.2f}x"),
            ColumnSpec("speedup_cuda", "CUDA/PT", lambda v: f"{float(v):.2f}x"),
            ColumnSpec("max_diff", "Max Diff TRT", lambda v: f"{float(v):.2e}"),
            ColumnSpec("max_diff_cuda", "Max Diff CUDA", lambda v: f"{float(v):.2e}"),
        ),
    )
)
