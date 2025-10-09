#!/usr/bin/env -S uv run python
"""Benchmark RTU (low-rank) Triton vs PyTorch implementations.

This script compares the performance of the Triton-accelerated RTU kernel
against the PyTorch reference across various configurations.

It measures forward runtime on CUDA (with warmup to amortize JIT) and reports
the max absolute difference between outputs for sanity.
"""

from __future__ import annotations

import time
from typing import Optional

import torch
from cortex.kernels.pytorch.rtu import LinearRTU as LinearRTU_PT

try:
    from cortex.kernels.triton import LinearRTU_Triton as LinearRTU_TR  # type: ignore
except Exception:  # pragma: no cover - allow running on CPU without Triton
    LinearRTU_TR = None  # type: ignore[assignment]


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
    """Benchmark RTU implementations.

    Args:
        batch_size: Batch size (B)
        seq_len: Sequence length (T)
        hidden_size: Hidden/input size (D == H)
        rank: Low-rank size (R)
        activation: Activation name (e.g., "SiLU", "ReLU", "Tanh")
        with_resets: Whether to include per-timestep resets
        reset_prob: Probability of reset at each time step per batch element
        num_warmup: Warmup iterations (JIT + caches)
        num_iterations: Timed iterations
        device: Device string; Triton path requires CUDA

    Returns:
        Dictionary with timing results and max difference
    """
    dtype = torch.float32
    device_t = torch.device(device)

    # Inputs: x in R^{B,T,H}; initial state zeros; optional resets
    x = torch.randn(batch_size, seq_len, hidden_size, device=device_t, dtype=dtype)
    h0 = torch.zeros(batch_size, hidden_size, device=device_t, dtype=dtype)
    c0 = torch.zeros(batch_size, hidden_size, device=device_t, dtype=dtype)
    resets = None
    if with_resets:
        resets = torch.rand(batch_size, seq_len, device=device_t) < reset_prob

    # Modules
    act = getattr(torch.nn, activation)() if hasattr(torch.nn, activation) else torch.nn.SiLU()
    pt = LinearRTU_PT(
        input_size=hidden_size,
        hidden_size=hidden_size,
        rank=rank,
        batch_first=True,
        activation=act,
    ).to(device=device_t, dtype=dtype)

    results: dict[str, Optional[float]] = {}

    # PyTorch baseline
    print("  Benchmarking PyTorch implementation...")
    for _ in range(num_warmup):
        _y_pt, _ = pt(x, (h0, c0), resets=resets)

    if device_t.type == "cuda" and LinearRTU_TR is not None:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_iterations):
        y_pt, _ = pt(x, (h0, c0), resets=resets)
    if device_t.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    pytorch_time = (t1 - t0) / num_iterations
    results["pytorch_ms"] = pytorch_time * 1000.0

    # Triton accelerated
    if device_t.type == "cuda":
        print("  Benchmarking Triton implementation...")
        tr = LinearRTU_TR(
            input_size=hidden_size,
            hidden_size=hidden_size,
            rank=rank,
            batch_first=True,
            activation=act,
        ).to(device=device_t, dtype=dtype)

        # Align parameters for a meaningful max-diff comparison
        with torch.no_grad():
            tr.nu_log.copy_(pt.nu_log)
            tr.theta_log.copy_(pt.theta_log)
            tr.U1.copy_(pt.U1)
            tr.U2.copy_(pt.U2)
            tr.V1.copy_(pt.V1)
            tr.V2.copy_(pt.V2)

        for _ in range(num_warmup):
            _y_tr, _ = tr(x, (h0, c0), resets=resets)
        torch.cuda.synchronize()

        t2 = time.perf_counter()
        for _ in range(num_iterations):
            y_tr, _ = tr(x, (h0, c0), resets=resets)
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

    print("Configuration format: (batch, seq_len, hidden, rank, resets, p)")
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

    print()
    print("=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
