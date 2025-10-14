#!/usr/bin/env -S uv run python
"""Benchmark: PyTorch Linear vs Axons cell (same input/output dims).

This script compares a standard ``torch.nn.Linear(H, H)`` layer against the
``cortex.cells.axons.Axons`` cell on identical inputs of shape ``[B, T, H]``.

What it measures
----------------
- Forward+backward time (averaged over iterations)
- Tokens/sec throughput (B*T per iteration)
- Parameter counts for context (they differ by design)

Backends
--------
Axons can run on multiple backends. You can force the backend via
``--axon-backend {auto,pytorch,triton,cuda}``.

Examples
--------
uv run python packages/cortex/benchmarks/bench_linear_vs_axon.py \
  --batch 8 --seq 512 --hidden 512 --iters 30 --axon-backend auto

uv run python packages/cortex/benchmarks/bench_linear_vs_axon.py \
  --device cuda --axon-backend cuda --seq 256 --hidden 256
"""

from __future__ import annotations

import argparse
import contextlib
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterator

import torch
import torch.nn as nn
from cortex.cells.axons import Axons
from cortex.config import AxonsConfig


@dataclass
class BenchResult:
    ms_per_iter: float
    tokens_per_s: float


def _count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


@contextlib.contextmanager
def _force_axon_backend(which: str) -> Iterator[None]:
    """Temporarily override ``select_backend`` to force a specific Axons backend.

    which: one of {auto, pytorch, triton, cuda}
    """
    import cortex.cells.axons as cell_mod

    if which == "auto":
        yield
        return

    orig = cell_mod.select_backend

    def chooser(*, triton_fn=None, pytorch_fn=None, tensor=None, allow_triton=True, cuda_fn=None, allow_cuda=False):  # type: ignore[override]
        if which == "pytorch":
            return pytorch_fn
        if which == "triton":
            return triton_fn
        if which == "cuda":
            return cuda_fn
        return orig(
            triton_fn=triton_fn,
            pytorch_fn=pytorch_fn,
            tensor=tensor,
            allow_triton=allow_triton,
            cuda_fn=cuda_fn,
            allow_cuda=allow_cuda,
        )

    try:
        cell_mod.select_backend = chooser  # type: ignore[assignment]
        yield
    finally:
        cell_mod.select_backend = orig  # type: ignore[assignment]


def _benchmark_model(
    fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    x: torch.Tensor,
    target: torch.Tensor,
    num_warmup: int,
    num_iterations: int,
    device: torch.device,
    params: tuple[torch.nn.Parameter, ...],
) -> BenchResult:
    # Warmup
    for _ in range(num_warmup):
        y = fn(x)
        loss = torch.nn.functional.mse_loss(y, target)
        loss.backward()
        for p in params:
            if p.grad is not None:
                p.grad = None

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(num_iterations):
        y = fn(x)
        loss = torch.nn.functional.mse_loss(y, target)
        loss.backward()
        # zero grads without optimizer to keep overhead minimal
        for p in params:
            if p.grad is not None:
                p.grad = None

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    ms = (t1 - t0) / num_iterations * 1000.0
    B, T, _ = x.shape
    tokens_per_s = (B * T) / ((t1 - t0) / num_iterations)
    return BenchResult(ms_per_iter=ms, tokens_per_s=tokens_per_s)


def benchmark_linear_vs_axon(
    *,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    activation: str,
    device: str,
    num_warmup: int,
    num_iterations: int,
    with_resets: bool,
    reset_prob: float,
    axon_backend: str,
    use_srht: bool,
    axon_out_rank: int,
) -> Dict[str, float | int]:
    device_t = torch.device(device)
    dtype = torch.float32

    # Inputs / targets
    x = torch.randn(batch_size, seq_len, hidden_size, device=device_t, dtype=dtype)
    target = torch.randn_like(x)
    resets = None
    if with_resets:
        resets = torch.rand(batch_size, seq_len, device=device_t) < reset_prob

    # Linear baseline
    lin = nn.Linear(hidden_size, hidden_size, bias=True).to(device=device_t, dtype=dtype)

    def run_linear(inp: torch.Tensor) -> torch.Tensor:
        Bt = inp.shape[0] * inp.shape[1]
        y = lin(inp.reshape(Bt, hidden_size))
        return y.reshape(inp.shape[0], inp.shape[1], hidden_size)

    # Axons cell
    cfg = AxonsConfig(
        hidden_size=hidden_size,
        activation=activation,
        cuda_seq_threshold=1000,
        use_srht=use_srht,
        out_dim=hidden_size,
        out_rank=axon_out_rank,
    )
    ax = Axons(cfg).to(device=device_t, dtype=dtype)

    def run_axon(inp: torch.Tensor) -> torch.Tensor:
        y, _ = ax(inp, state=None, resets=resets)
        return y

    # Count params
    n_lin = _count_parameters(lin)
    n_ax = _count_parameters(ax)

    # Benchmark
    if device_t.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with _force_axon_backend(axon_backend):
        res_lin = _benchmark_model(
            run_linear,
            x=x,
            target=target,
            num_warmup=num_warmup,
            num_iterations=num_iterations,
            device=device_t,
            params=tuple(lin.parameters()),
        )
        res_ax = _benchmark_model(
            run_axon,
            x=x,
            target=target,
            num_warmup=num_warmup,
            num_iterations=num_iterations,
            device=device_t,
            params=tuple(ax.parameters()),
        )

    peak_mem = torch.cuda.max_memory_allocated() if device_t.type == "cuda" else 0

    return {
        "linear_ms": res_lin.ms_per_iter,
        "axon_ms": res_ax.ms_per_iter,
        "speedup": res_lin.ms_per_iter / res_ax.ms_per_iter,
        "linear_toks_s": res_lin.tokens_per_s,
        "axon_toks_s": res_ax.tokens_per_s,
        "linear_params": n_lin,
        "axon_params": n_ax,
        "peak_mem_bytes": peak_mem,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark PyTorch Linear vs Axons cell")
    p.add_argument("--batch", type=int, default=8, help="Batch size B")
    p.add_argument("--seq", type=int, default=512, help="Sequence length T")
    p.add_argument("--hidden", type=int, default=512, help="Hidden size H (in==out)")
    p.add_argument("--activation", type=str, default="identity", help="Axons activation (silu|relu|tanh|identity)")
    p.add_argument(
        "--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"), help="Device: cuda or cpu"
    )
    p.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    p.add_argument("--iters", type=int, default=50, help="Measured iterations")
    p.add_argument("--with-resets", action="store_true", help="Enable stochastic resets for Axons")
    p.add_argument("--reset-prob", type=float, default=0.1, help="Reset probability per timestep if --with-resets")
    p.add_argument(
        "--axon-backend",
        type=str,
        choices=["auto", "pytorch", "triton", "cuda"],
        default="auto",
        help="Force Axons backend (auto=default policy)",
    )
    p.add_argument("--no-srht", dest="use_srht", action="store_false", help="Disable SRHT mixer (enabled by default)")
    p.set_defaults(use_srht=True)
    p.add_argument(
        "--axon-out-rank",
        type=int,
        default=-1,
        help="Low-rank projection for Axons (2H->r->H). Default uses H//2.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        device = "cpu"

    print("=" * 80)
    print("PyTorch Linear vs Axons Benchmark")
    print("=" * 80)
    if device == "cuda":
        print(f"Device: {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda}")
    else:
        print("Device: CPU")
    out_rank = args.axon_out_rank if args.axon_out_rank > 0 else args.hidden // 2
    print(
        f"Config: B={args.batch}, T={args.seq}, H={args.hidden}, activation={args.activation}, "
        f"backend={args.axon_backend}, SRHT={'on' if args.use_srht else 'off'}, out_rank={out_rank}"
    )
    print(f"Warmup={args.warmup}, Iters={args.iters}")
    print()

    res = benchmark_linear_vs_axon(
        batch_size=args.batch,
        seq_len=args.seq,
        hidden_size=args.hidden,
        activation=args.activation,
        device=device,
        num_warmup=args.warmup,
        num_iterations=args.iters,
        with_resets=args.with_resets,
        reset_prob=args.reset_prob,
        axon_backend=args.axon_backend,
        use_srht=args.use_srht,
        axon_out_rank=out_rank,
    )

    print(f"Linear (ms): {res['linear_ms']:.3f}")
    print(f"Axons  (ms): {res['axon_ms']:.3f}")
    print(f"Speedup   : {res['speedup']:.2f}x (Linear/Axons)")
    print(f"Throughput: linear={res['linear_toks_s']:.1f} tok/s, axons={res['axon_toks_s']:.1f} tok/s")
    print(f"Params    : linear={int(res['linear_params'])}, axons={int(res['axon_params'])}")
    if args.device == "cuda":
        gb = res["peak_mem_bytes"] / (1024**3)
        print(f"Peak CUDA memory: {gb:.2f} GiB")

    print("\nDone.")


if __name__ == "__main__":
    main()
