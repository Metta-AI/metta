#!/usr/bin/env -S uv run python
"""Benchmark Conv1D Triton vs PyTorch implementations.

This script compares the performance of Triton-accelerated causal conv1d kernels
against pure PyTorch implementations across various configurations.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
from cortex.kernels.pytorch.conv1d import causal_conv1d_pytorch
from cortex.kernels.triton.conv1d import causal_conv1d_triton


def benchmark_conv1d(
    batch_size: int,
    seq_len: int,
    features: int,
    kernel_size: int,
    num_warmup: int = 10,
    num_iterations: int = 50,
    device: str = "cuda",
) -> dict[str, float]:
    """Benchmark Conv1D implementations.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        features: Number of features/channels
        kernel_size: Convolution kernel size
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations
        device: Device to run on

    Returns:
        Dictionary with timing results
    """
    dtype = torch.float32
    device = torch.device(device)

    # Create random inputs
    x = torch.randn(batch_size, seq_len, features, device=device, dtype=dtype)
    conv_state = torch.randn(batch_size, kernel_size, features, device=device, dtype=dtype)
    weight = torch.randn(features, features, kernel_size, device=device, dtype=dtype)
    bias = torch.randn(features, device=device, dtype=dtype)
    resets = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)  # No resets for fair comparison

    groups = 1  # Channel-mixing mode for Triton
    pad = kernel_size - 1

    # Create conv module for PyTorch fast path
    conv = nn.Conv1d(
        in_channels=features,
        out_channels=features,
        kernel_size=kernel_size,
        padding=pad,
        groups=groups,
        bias=True,
        device=device,
        dtype=dtype,
    )
    # Copy weights to match Triton kernel
    with torch.no_grad():
        conv.weight.copy_(weight)
        conv.bias.copy_(bias)

    results = {}

    # Benchmark PyTorch implementation (without resets for fair comparison)
    print("  Benchmarking PyTorch implementation...")
    for _ in range(num_warmup):
        _ = causal_conv1d_pytorch(
            conv_state=conv_state.clone(),
            x=x,
            weight=weight,
            bias=bias,
            groups=groups,
            pad=pad,
            conv=conv,
            resets=None,  # No resets for PyTorch fast path
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(num_iterations):
        y_pytorch, state_pytorch = causal_conv1d_pytorch(
            conv_state=conv_state.clone(),
            x=x,
            weight=weight,
            bias=bias,
            groups=groups,
            pad=pad,
            conv=conv,
            resets=None,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    pytorch_time = (end_time - start_time) / num_iterations
    results["pytorch_ms"] = pytorch_time * 1000

    # Benchmark Triton implementation (requires resets parameter)
    if device.type == "cuda":
        print("  Benchmarking Triton implementation...")
        for _ in range(num_warmup):
            try:
                _ = causal_conv1d_triton(
                    conv_state=conv_state.clone(),
                    x=x,
                    weight=weight,
                    bias=bias,
                    groups=groups,
                    resets=resets,
                )
            except Exception:
                pass

        try:
            torch.cuda.synchronize()

            start_time = time.perf_counter()
            for _ in range(num_iterations):
                y_triton, state_triton = causal_conv1d_triton(
                    conv_state=conv_state.clone(),
                    x=x,
                    weight=weight,
                    bias=bias,
                    groups=groups,
                    resets=resets,
                )
            torch.cuda.synchronize()
            end_time = time.perf_counter()

            triton_time = (end_time - start_time) / num_iterations
            results["triton_ms"] = triton_time * 1000
            results["speedup"] = pytorch_time / triton_time

            # Verify outputs match (approximately, due to different computation paths)
            max_diff = torch.max(torch.abs(y_pytorch - y_triton)).item()
            results["max_diff"] = max_diff
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
    print("Conv1D Triton vs PyTorch Benchmark")
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
    configs = [
        # (batch_size, seq_len, features, kernel_size)
        (2, 128, 256, 4),
        (2, 256, 256, 4),
        (2, 512, 256, 4),
        (4, 128, 512, 4),
        (4, 256, 512, 4),
        (4, 512, 512, 4),
        (8, 256, 512, 4),
        (8, 512, 512, 4),
        (8, 1024, 512, 4),
        (4, 512, 512, 8),
        (8, 512, 512, 8),
    ]

    print("Configuration format: (batch, seq_len, features, kernel_size)")
    print("Note: Triton kernel is optimized for channel-mixing mode with per-timestep resets")
    print()
    print(f"{'Config':<35} {'PyTorch (ms)':<15} {'Triton (ms)':<15} {'Speedup':<10} {'Max Diff':<12}")
    print("-" * 95)

    for batch_size, seq_len, features, kernel_size in configs:
        config_str = f"({batch_size}, {seq_len}, {features}, {kernel_size})"
        print(f"{config_str:<35}", end=" ", flush=True)

        try:
            results = benchmark_conv1d(
                batch_size=batch_size,
                seq_len=seq_len,
                features=features,
                kernel_size=kernel_size,
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
