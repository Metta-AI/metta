#!/usr/bin/env python3
"""Benchmark sLSTM Triton vs PyTorch implementations.

This script compares the performance of Triton-accelerated sLSTM kernels
against pure PyTorch implementations across various configurations.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

# Add cortex to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cortex.kernels.pytorch.slstm import slstm_sequence_pytorch
from cortex.kernels.triton.slstm import slstm_sequence_triton


def benchmark_slstm(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    num_warmup: int = 10,
    num_iterations: int = 50,
    device: str = "cuda",
) -> dict[str, float]:
    """Benchmark sLSTM implementations.

    Args:
        batch_size: Batch size
        num_heads: Number of heads
        seq_len: Sequence length
        head_dim: Head dimension (must be power of 2 for Triton)
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations
        device: Device to run on

    Returns:
        Dictionary with timing results
    """
    dtype = torch.float32
    device = torch.device(device)

    # Create random inputs
    Wx = torch.randn(batch_size, seq_len, 4, num_heads, head_dim, device=device, dtype=dtype)
    R = torch.randn(4, num_heads, head_dim, head_dim, device=device, dtype=dtype)
    b = torch.randn(4, num_heads, head_dim, device=device, dtype=dtype)
    initial_states = torch.randn(4, batch_size, num_heads, head_dim, device=device, dtype=dtype)

    results = {}

    # Benchmark PyTorch implementation
    print("  Benchmarking PyTorch implementation...")
    for _ in range(num_warmup):
        _ = slstm_sequence_pytorch(
            Wx=Wx,
            R=R,
            b=b,
            initial_states=initial_states,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(num_iterations):
        all_states_pytorch, last_state_pytorch = slstm_sequence_pytorch(
            Wx=Wx,
            R=R,
            b=b,
            initial_states=initial_states,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    pytorch_time = (end_time - start_time) / num_iterations
    results["pytorch_ms"] = pytorch_time * 1000

    # Benchmark Triton implementation
    if device.type == "cuda":
        print("  Benchmarking Triton implementation...")
        for _ in range(num_warmup):
            _ = slstm_sequence_triton(
                Wx=Wx,
                R=R,
                b=b,
                initial_states=initial_states,
            )

        torch.cuda.synchronize()

        start_time = time.perf_counter()
        for _ in range(num_iterations):
            all_states_triton, last_state_triton = slstm_sequence_triton(
                Wx=Wx,
                R=R,
                b=b,
                initial_states=initial_states,
            )
        torch.cuda.synchronize()
        end_time = time.perf_counter()

        triton_time = (end_time - start_time) / num_iterations
        results["triton_ms"] = triton_time * 1000
        results["speedup"] = pytorch_time / triton_time

        # Verify outputs match
        max_diff_states = torch.max(torch.abs(all_states_pytorch - all_states_triton)).item()
        max_diff_last = torch.max(torch.abs(last_state_pytorch - last_state_triton)).item()

        results["max_diff"] = max(max_diff_states, max_diff_last)
    else:
        results["triton_ms"] = None
        results["speedup"] = None
        results["max_diff"] = None
        print("  Skipping Triton (CUDA not available)")

    return results


def main():
    """Run benchmark suite."""
    print("=" * 80)
    print("sLSTM Triton vs PyTorch Benchmark")
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
    # Note: head_dim must be power of 2 for Triton
    configs = [
        # (batch_size, num_heads, seq_len, head_dim)
        (2, 4, 128, 64),
        (2, 4, 256, 64),
        (2, 4, 512, 64),
        (4, 8, 128, 64),
        (4, 8, 256, 64),
        (4, 8, 512, 64),
        (8, 8, 256, 64),
        (8, 8, 512, 64),
        (8, 8, 1024, 64),
        (4, 4, 512, 128),
        (8, 4, 512, 128),
    ]

    print("Configuration format: (batch, heads, seq_len, head_dim)")
    print()
    print(f"{'Config':<35} {'PyTorch (ms)':<15} {'Triton (ms)':<15} {'Speedup':<10} {'Max Diff':<12}")
    print("-" * 95)

    for batch_size, num_heads, seq_len, head_dim in configs:
        config_str = f"({batch_size}, {num_heads}, {seq_len}, {head_dim})"
        print(f"{config_str:<35}", end=" ", flush=True)

        try:
            results = benchmark_slstm(
                batch_size=batch_size,
                num_heads=num_heads,
                seq_len=seq_len,
                head_dim=head_dim,
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
