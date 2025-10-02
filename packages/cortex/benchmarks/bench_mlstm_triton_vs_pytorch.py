#!/usr/bin/env python3
"""Benchmark mLSTM Triton vs PyTorch implementations.

This script compares the performance of Triton-accelerated mLSTM kernels
against pure PyTorch implementations across various configurations.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

# Add cortex to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cortex.kernels.pytorch.mlstm import mlstm_chunkwise_simple
from cortex.kernels.triton.mlstm import mlstm_chunkwise_triton


def benchmark_mlstm(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    chunk_size: int = 64,
    num_warmup: int = 10,
    num_iterations: int = 50,
    device: str = "cuda",
) -> dict[str, float]:
    """Benchmark mLSTM implementations.

    Args:
        batch_size: Batch size
        num_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Head dimension
        chunk_size: Chunk size for chunkwise processing
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations
        device: Device to run on

    Returns:
        Dictionary with timing results
    """
    dtype = torch.float32
    device = torch.device(device)

    # Create random inputs
    queries = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    keys = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    values = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    igate_preact = torch.randn(batch_size, num_heads, seq_len, device=device, dtype=dtype)
    fgate_preact = torch.randn(batch_size, num_heads, seq_len, device=device, dtype=dtype)

    results = {}

    # Benchmark PyTorch implementation
    print(f"  Benchmarking PyTorch implementation...")
    for _ in range(num_warmup):
        _ = mlstm_chunkwise_simple(
            queries=queries,
            keys=keys,
            values=values,
            igate_preact=igate_preact,
            fgate_preact=fgate_preact,
            chunk_size=chunk_size,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(num_iterations):
        output_pytorch = mlstm_chunkwise_simple(
            queries=queries,
            keys=keys,
            values=values,
            igate_preact=igate_preact,
            fgate_preact=fgate_preact,
            chunk_size=chunk_size,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    pytorch_time = (end_time - start_time) / num_iterations
    results["pytorch_ms"] = pytorch_time * 1000

    # Benchmark Triton implementation
    if device.type == "cuda":
        print(f"  Benchmarking Triton implementation...")
        for _ in range(num_warmup):
            _ = mlstm_chunkwise_triton(
                queries=queries,
                keys=keys,
                values=values,
                igate_preact=igate_preact,
                fgate_preact=fgate_preact,
                chunk_size=chunk_size,
            )

        torch.cuda.synchronize()

        start_time = time.perf_counter()
        for _ in range(num_iterations):
            output_triton = mlstm_chunkwise_triton(
                queries=queries,
                keys=keys,
                values=values,
                igate_preact=igate_preact,
                fgate_preact=fgate_preact,
                chunk_size=chunk_size,
            )
        torch.cuda.synchronize()
        end_time = time.perf_counter()

        triton_time = (end_time - start_time) / num_iterations
        results["triton_ms"] = triton_time * 1000
        results["speedup"] = pytorch_time / triton_time

        # Verify outputs match
        max_diff = torch.max(torch.abs(output_pytorch - output_triton)).item()
        results["max_diff"] = max_diff
    else:
        results["triton_ms"] = None
        results["speedup"] = None
        results["max_diff"] = None
        print(f"  Skipping Triton (CUDA not available)")

    return results


def main():
    """Run benchmark suite."""
    print("=" * 80)
    print("mLSTM Triton vs PyTorch Benchmark")
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
        # (batch_size, num_heads, seq_len, head_dim, chunk_size)
        (2, 4, 128, 64, 64),
        (2, 4, 256, 64, 64),
        (2, 4, 512, 64, 64),
        (4, 8, 128, 64, 64),
        (4, 8, 256, 64, 64),
        (4, 8, 512, 64, 64),
        (8, 8, 256, 64, 64),
        (8, 8, 512, 64, 64),
        (8, 8, 1024, 64, 64),
    ]

    print("Configuration format: (batch, heads, seq_len, head_dim, chunk_size)")
    print()
    print(f"{'Config':<40} {'PyTorch (ms)':<15} {'Triton (ms)':<15} {'Speedup':<10} {'Max Diff':<12}")
    print("-" * 100)

    for batch_size, num_heads, seq_len, head_dim, chunk_size in configs:
        config_str = f"({batch_size}, {num_heads}, {seq_len}, {head_dim}, {chunk_size})"
        print(f"{config_str:<40}", end=" ", flush=True)

        try:
            results = benchmark_mlstm(
                batch_size=batch_size,
                num_heads=num_heads,
                seq_len=seq_len,
                head_dim=head_dim,
                chunk_size=chunk_size,
                num_warmup=5,
                num_iterations=20,
                device=device,
            )

            pytorch_str = f"{results['pytorch_ms']:.3f}"
            if results["triton_ms"] is not None:
                triton_str = f"{results['triton_ms']:.3f}"
                speedup_str = f"{results['speedup']:.2f}x"
                diff_str = f"{results['max_diff']:.2e}"
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
