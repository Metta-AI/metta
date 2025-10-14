#!/usr/bin/env -S uv run python
"""Benchmark RTU (low-rank) Triton vs PyTorch using the RTUCell.

This compares the Triton-accelerated path vs the PyTorch path inside
`cortex.cells.rtu.RTUCell` by forcing backend selection via a small
monkeypatch of `select_backend`.

Additionally, for the diagonal streaming kernels (D == H), this script
benchmarks the PyTorch functional, the Triton implementation, and the new
CUDA “sequential all-in” kernel side-by-side.
"""

from __future__ import annotations

import time
from typing import Optional

import torch
from cortex.cells.rtu import RTUCell
from cortex.config import RTUCellConfig


def _run_cell(cell: RTUCell, x: torch.Tensor, resets: Optional[torch.Tensor], which: str) -> torch.Tensor:
    import cortex.cells.rtu as cell_mod

    orig = cell_mod.select_backend

    def chooser(*, triton_fn, pytorch_fn, tensor, allow_triton=True):  # type: ignore[override]
        return triton_fn if which == "triton" else pytorch_fn

    try:
        cell_mod.select_backend = chooser  # type: ignore[assignment]
        y, _ = cell(x, state=None, resets=resets)
        return y
    finally:
        cell_mod.select_backend = orig  # type: ignore[assignment]


@torch.inference_mode(False)
def benchmark_rtu(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    rank: int,
    *,
    activation: str = "SiLU",
    with_resets: bool = False,
    reset_prob: float = 0.0,
    num_warmup: int = 10,
    num_iterations: int = 50,
    device: str = "cuda",
) -> dict[str, Optional[float]]:
    dtype = torch.float32
    device_t = torch.device(device)

    # Inputs: x in R^{B,T,H}; optional resets
    x = torch.randn(batch_size, seq_len, hidden_size, device=device_t, dtype=dtype)
    resets = None
    if with_resets:
        resets = torch.rand(batch_size, seq_len, device=device_t) < reset_prob

    # Build a single cell whose parameters are shared across both backends
    cell = RTUCell(RTUCellConfig(hidden_size=hidden_size, rank=rank, activation=activation)).to(
        device=device_t, dtype=dtype
    )

    results: dict[str, Optional[float]] = {}

    # PyTorch baseline
    print("  Benchmarking PyTorch implementation...")
    for _ in range(num_warmup):
        _ = _run_cell(cell, x, resets, which="pytorch")

    if device_t.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_iterations):
        y_pt = _run_cell(cell, x, resets, which="pytorch")
    if device_t.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    pytorch_time = (t1 - t0) / num_iterations
    results["pytorch_ms"] = pytorch_time * 1000.0

    # Triton accelerated
    if device_t.type == "cuda":
        print("  Benchmarking Triton implementation...")
        for _ in range(num_warmup):
            _ = _run_cell(cell, x, resets, which="triton")
        torch.cuda.synchronize()

        t2 = time.perf_counter()
        for _ in range(num_iterations):
            y_tr = _run_cell(cell, x, resets, which="triton")
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        triton_time = (t3 - t2) / num_iterations

        results["triton_ms"] = triton_time * 1000.0
        results["speedup"] = pytorch_time / triton_time
        results["max_diff"] = torch.max(torch.abs(y_pt - y_tr)).item()
    else:
        results["triton_ms"] = None
        results["speedup"] = None
        results["max_diff"] = None
        print("  Skipping Triton (CUDA not available or Triton not installed)")

    return results


# -----------------------------
# Streaming (diagonal, D == H)
# -----------------------------


def _maybe_stream_kernels():
    pt = None
    tri = None
    cu = None
    # PyTorch reference
    try:
        from cortex.kernels.pytorch.rtu_stream import (
            rtu_stream_diag_pytorch as pt_fn,
        )

        pt = pt_fn
    except Exception:
        pass
    # Triton (optional)
    try:
        from cortex.kernels.triton import (
            rtu_stream_diag_triton as tri_fn,
        )

        tri = tri_fn
    except Exception:
        tri = None
    # CUDA seq-allin (optional)
    try:
        from cortex.kernels.cuda import (
            rtu_stream_diag_cuda_seq_allin as cu_fn,
        )

        cu = cu_fn
    except Exception:
        cu = None
    return pt, tri, cu


@torch.inference_mode(False)
def benchmark_rtu_stream_diag(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    *,
    activation: str = "SiLU",
    with_resets: bool = False,
    reset_prob: float = 0.0,
    num_warmup: int = 10,
    num_iterations: int = 50,
    device: str = "cuda",
) -> dict[str, Optional[float]]:
    device_t = torch.device(device)
    dtype = torch.float32

    pt, tri, cu = _maybe_stream_kernels()
    if pt is None:
        raise RuntimeError("PyTorch streaming diag kernel not available")

    B, T, H = batch_size, seq_len, hidden_size
    D = H
    x = torch.randn(B, T, D, device=device_t, dtype=dtype)
    nu_log = torch.randn(H, device=device_t, dtype=dtype)
    theta_log = torch.randn(H, device=device_t, dtype=dtype)
    w1 = torch.randn(H, device=device_t, dtype=dtype)
    w2 = torch.randn(H, device=device_t, dtype=dtype)
    hc1 = torch.zeros(B, H, device=device_t, dtype=dtype)
    hc2 = torch.zeros(B, H, device=device_t, dtype=dtype)
    trace_in = None
    resets = None
    if with_resets:
        resets = torch.rand(B, T, device=device_t) < reset_prob

    results: dict[str, Optional[float]] = {}

    # Warmup PT
    for _ in range(num_warmup):
        y_pt, _, _ = pt(
            x_btd=x,
            nu_log=nu_log,
            theta_log=theta_log,
            w1=w1,
            w2=w2,
            activation_name=activation,
            hc1_init_bh=hc1,
            hc2_init_bh=hc2,
            trace_in=trace_in,
            resets_bt=resets,
        )
    if device_t.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_iterations):
        y_pt, _, _ = pt(
            x_btd=x,
            nu_log=nu_log,
            theta_log=theta_log,
            w1=w1,
            w2=w2,
            activation_name=activation,
            hc1_init_bh=hc1,
            hc2_init_bh=hc2,
            trace_in=trace_in,
            resets_bt=resets,
        )
    if device_t.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    pt_ms = (t1 - t0) / num_iterations * 1000.0
    results["pt_ms"] = pt_ms

    # Triton (optional)
    if device_t.type == "cuda" and tri is not None:
        for _ in range(max(2, num_warmup)):
            y_tr, _, _ = tri(
                x_btd=x,
                nu_log=nu_log,
                theta_log=theta_log,
                w1=w1,
                w2=w2,
                activation_name=activation,
                hc1_init_bh=hc1,
                hc2_init_bh=hc2,
                trace_in=trace_in,
                resets_bt=resets,
            )
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        for _ in range(num_iterations):
            y_tr, _, _ = tri(
                x_btd=x,
                nu_log=nu_log,
                theta_log=theta_log,
                w1=w1,
                w2=w2,
                activation_name=activation,
                hc1_init_bh=hc1,
                hc2_init_bh=hc2,
                trace_in=trace_in,
                resets_bt=resets,
            )
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        tri_ms = (t3 - t2) / num_iterations * 1000.0
        results["tri_ms"] = tri_ms
        results["tri_speedup_vs_pt"] = pt_ms / tri_ms
        results["tri_diff_vs_pt"] = torch.max(torch.abs(y_pt - y_tr)).item()
    else:
        results["tri_ms"] = None
        results["tri_speedup_vs_pt"] = None
        results["tri_diff_vs_pt"] = None

    # CUDA seq-allin (optional)
    if device_t.type == "cuda" and cu is not None:
        for _ in range(max(2, num_warmup)):
            y_cu, _, _ = cu(
                x_btd=x,
                nu_log=nu_log,
                theta_log=theta_log,
                w1=w1,
                w2=w2,
                activation_name=activation,
                hc1_init_bh=hc1,
                hc2_init_bh=hc2,
                trace_in=trace_in,
                resets_bt=resets,
            )
        torch.cuda.synchronize()
        t4 = time.perf_counter()
        for _ in range(num_iterations):
            y_cu, _, _ = cu(
                x_btd=x,
                nu_log=nu_log,
                theta_log=theta_log,
                w1=w1,
                w2=w2,
                activation_name=activation,
                hc1_init_bh=hc1,
                hc2_init_bh=hc2,
                trace_in=trace_in,
                resets_bt=resets,
            )
        torch.cuda.synchronize()
        t5 = time.perf_counter()
        cu_ms = (t5 - t4) / num_iterations * 1000.0
        results["cuda_ms"] = cu_ms
        results["cuda_speedup_vs_pt"] = pt_ms / cu_ms
        results["cuda_diff_vs_pt"] = torch.max(torch.abs(y_pt - y_cu)).item()
    else:
        results["cuda_ms"] = None
        results["cuda_speedup_vs_pt"] = None
        results["cuda_diff_vs_pt"] = None

    return results


def main() -> None:
    print("=" * 80)
    print("RTU (low-rank) Triton vs PyTorch Benchmark")
    print("=" * 80)
    print()

    if torch.cuda.is_available():
        device = "cuda"
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        device = "cpu"
        print("Device: CPU (Triton benchmarks will be skipped)")
    print()

    # (batch, seq_len, hidden, rank, resets?, p)
    configs = [
        (4, 128, 64, 8, False, 0.0),
        (4, 256, 64, 8, False, 0.0),
        (8, 256, 64, 16, False, 0.0),
        (8, 512, 64, 16, False, 0.0),
        (8, 512, 128, 16, False, 0.0),
        # Resets on: segmented scan path
        (8, 512, 64, 16, True, 0.1),
        (8, 1024, 64, 16, True, 0.1),
    ]

    print("Configuration format (low-rank RTU): (batch, seq_len, hidden, rank, resets, p)")
    print()
    print(f"{'Config':<48} {'PyTorch (ms)':<15} {'Triton (ms)':<15} {'Speedup':<10} {'Max Diff':<12}")
    print("-" * 110)

    for b, t, h, r, use_resets, p in configs:
        cfg = f"({b}, {t}, {h}, {r}, {use_resets}, {p})"
        print(f"{cfg:<48}", end=" ", flush=True)
        try:
            res = benchmark_rtu(
                batch_size=b,
                seq_len=t,
                hidden_size=h,
                rank=r,
                with_resets=use_resets,
                reset_prob=p,
                num_warmup=5,
                num_iterations=20,
                device=device,
            )

            pyt = f"{res['pytorch_ms']:.3f}"
            if res["triton_ms"] is not None:
                tri = f"{res['triton_ms']:.3f}"
                spd = f"{res['speedup']:.2f}x"
                diff = f"{res['max_diff']:.2e}"
            else:
                tri = "N/A"
                spd = "N/A"
                diff = "N/A"
            print(f"{pyt:<15} {tri:<15} {spd:<10} {diff:<12}")
        except Exception as e:
            print(f"ERROR: {e}")

    # ------------------------------
    # Streaming diag (D == H) block
    # ------------------------------
    print()
    print("Streaming Diag (D == H) — PyTorch vs Triton vs CUDA (seq-allin)")
    print("Configuration format (diag RTU): (batch, seq_len, hidden, resets, p)")
    diag_cfgs = [
        (4, 128, 64, False, 0.0),
        (8, 256, 64, False, 0.0),
        (8, 512, 64, True, 0.1),
    ]
    print(
        f"{'Config':<36} {'PT (ms)':<10} {'TRI (ms)':<10} {'CU (ms)':<10} "
        f"{'TRI xPT':<8} {'CU xPT':<8} {'TRI|PT Δ':<10} {'CU|PT Δ':<10}"
    )
    print("-" * 110)
    for b, t, h, use_resets, p in diag_cfgs:
        cfg = f"({b}, {t}, {h}, {use_resets}, {p})"
        print(f"{cfg:<36}", end=" ", flush=True)
        try:
            res = benchmark_rtu_stream_diag(
                batch_size=b,
                seq_len=t,
                hidden_size=h,
                with_resets=use_resets,
                reset_prob=p,
                num_warmup=5,
                num_iterations=20,
                device=device,
            )
            pt_ms = f"{res['pt_ms']:.3f}"
            tri_ms = "N/A" if res["tri_ms"] is None else f"{res['tri_ms']:.3f}"
            cu_ms = "N/A" if res["cuda_ms"] is None else f"{res['cuda_ms']:.3f}"
            spd_tri = "N/A" if res["tri_speedup_vs_pt"] is None else f"{res['tri_speedup_vs_pt']:.2f}x"
            spd_cu = "N/A" if res["cuda_speedup_vs_pt"] is None else f"{res['cuda_speedup_vs_pt']:.2f}x"
            diff_tri = "N/A" if res["tri_diff_vs_pt"] is None else f"{res['tri_diff_vs_pt']:.2e}"
            diff_cu = "N/A" if res["cuda_diff_vs_pt"] is None else f"{res['cuda_diff_vs_pt']:.2e}"
            print(f"{pt_ms:<10} {tri_ms:<10} {cu_ms:<10} {spd_tri:<8} {spd_cu:<8} {diff_tri:<10} {diff_cu:<10}")
        except Exception as e:
            print(f"ERROR: {e}")

    print()
    print("=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
