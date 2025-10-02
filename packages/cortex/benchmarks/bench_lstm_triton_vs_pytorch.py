#!/usr/bin/env python3
"""Benchmark LSTM Triton vs PyTorch implementations.

This script compares the performance of Triton-accelerated LSTM kernels
against pure PyTorch implementations across various configurations.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# Add cortex to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cortex.kernels.pytorch.lstm import lstm_sequence_pytorch
from cortex.kernels.triton.lstm import lstm_sequence_triton


def benchmark_lstm(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_warmup: int = 10,
    num_iterations: int = 50,
    device: str = "cuda",
) -> dict[str, float]:
    """Benchmark LSTM implementations.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        hidden_size: Hidden size (must be power of 2 for Triton)
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations
        device: Device to run on

    Returns:
        Dictionary with timing results
    """
    dtype = torch.float32
    device = torch.device(device)

    # Create LSTM module
    lstm = nn.LSTM(
        input_size=hidden_size,
        hidden_size=hidden_size,
        num_layers=1,
        bias=True,
        batch_first=True,
    ).to(device=device, dtype=dtype)

    # Create random inputs
    x_seq = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    h0_bf = torch.randn(batch_size, 1, hidden_size, device=device, dtype=dtype)
    c0_bf = torch.randn(batch_size, 1, hidden_size, device=device, dtype=dtype)

    results = {}

    # Benchmark PyTorch implementation
    print("  Benchmarking PyTorch implementation...")
    for _ in range(num_warmup):
        _ = lstm_sequence_pytorch(
            lstm=lstm,
            x_seq=x_seq,
            h0_bf=h0_bf,
            c0_bf=c0_bf,
            resets=None,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(num_iterations):
        y_pytorch, hn_pytorch, cn_pytorch = lstm_sequence_pytorch(
            lstm=lstm,
            x_seq=x_seq,
            h0_bf=h0_bf,
            c0_bf=c0_bf,
            resets=None,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    pytorch_time = (end_time - start_time) / num_iterations
    results["pytorch_ms"] = pytorch_time * 1000

    # Benchmark Triton implementation
    if device.type == "cuda":
        print("  Benchmarking Triton implementation...")
        try:
            for _ in range(num_warmup):
                _ = lstm_sequence_triton(
                    lstm=lstm,
                    x_seq=x_seq,
                    h0_bf=h0_bf,
                    c0_bf=c0_bf,
                    resets=None,
                )

            torch.cuda.synchronize()

            start_time = time.perf_counter()
            for _ in range(num_iterations):
                y_triton, hn_triton, cn_triton = lstm_sequence_triton(
                    lstm=lstm,
                    x_seq=x_seq,
                    h0_bf=h0_bf,
                    c0_bf=c0_bf,
                    resets=None,
                )
            torch.cuda.synchronize()
            end_time = time.perf_counter()

            triton_time = (end_time - start_time) / num_iterations
            results["triton_ms"] = triton_time * 1000
            results["speedup"] = pytorch_time / triton_time

            # Verify outputs match
            max_diff_y = torch.max(torch.abs(y_pytorch - y_triton)).item()
            max_diff_hn = torch.max(torch.abs(hn_pytorch - hn_triton)).item()
            max_diff_cn = torch.max(torch.abs(cn_pytorch - cn_triton)).item()
            results["max_diff"] = max(max_diff_y, max_diff_hn, max_diff_cn)
        except Exception as e:
            results["triton_ms"] = None
            results["speedup"] = None
            results["max_diff"] = None
            results["error"] = str(e)
    else:
        results["triton_ms"] = None
        results["speedup"] = None
        results["max_diff"] = None
        print("  Skipping Triton (CUDA not available)")

    return results


def main():
    """Run benchmark suite."""
    print("=" * 80)
    print("LSTM Triton vs PyTorch Benchmark")
    print("=" * 80)
    print()

    # Check if CUDA is available
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        device = "cpu"
        print("Device: CPU (Triton benchmarks will be skipped)")
    print()

    # Benchmark configurations
    # Note: hidden_size must be power of 2 for Triton, batch >= 16 for Triton matmul
    configs = [
        # (batch_size, seq_len, hidden_size)
        (16, 128, 64),
        (16, 256, 64),
        (16, 512, 64),
        (32, 128, 64),
        (32, 256, 64),
        (32, 512, 64),
        (16, 256, 128),
        (32, 256, 128),
        (16, 512, 128),
        (32, 512, 128),
    ]

    print("Configuration format: (batch, seq_len, hidden_size)")
    print("Note: Batch size >= 16 required for Triton kernel (matmul constraints)")
    print()
    print(f"{'Config':<35} {'PyTorch (ms)':<15} {'Triton (ms)':<15} {'Speedup':<10} {'Max Diff':<12}")
    print("-" * 95)

    for batch_size, seq_len, hidden_size in configs:
        config_str = f"({batch_size}, {seq_len}, {hidden_size})"
        print(f"{config_str:<35}", end=" ", flush=True)

        try:
            results = benchmark_lstm(
                batch_size=batch_size,
                seq_len=seq_len,
                hidden_size=hidden_size,
                num_warmup=5,
                num_iterations=20,
                device=device,
            )

            pytorch_str = f"{results['pytorch_ms']:.3f}"
            if results["triton_ms"] is not None:
                triton_str = f"{results['triton_ms']:.3f}"
                speedup_str = f"{results['speedup']:.2f}x"
                diff_str = f"{results['max_diff']:.2e}"
            elif "error" in results:
                triton_str = "ERROR"
                speedup_str = "N/A"
                diff_str = "N/A"
                print()
                print(f"    Error: {results['error']}")
                continue
            else:
                triton_str = "N/A"
                speedup_str = "N/A"
                diff_str = "N/A"

            print(f"{pytorch_str:<15} {triton_str:<15} {speedup_str:<10} {diff_str:<12}")

        except Exception as e:
            print(f"ERROR: {str(e)}")

    print()
    print("=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
