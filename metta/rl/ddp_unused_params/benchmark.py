#!/usr/bin/env python3
"""Benchmark DDP unused parameter handling approaches.

This script compares three approaches for handling unused parameters in PyTorch DDP:
1. find_unused_parameters=True (PyTorch built-in, traverses autograd graph)
2. The 0.0 * value.sum() hack (manually touches unused outputs)
3. Baseline with all params used (no unused params, theoretical best case)

Usage Examples:

    # Quick test on CPU (2 processes)
    uv run torchrun --nproc-per-node=2 -m metta.rl.ddp_unused_params.benchmark \\
        --device=cpu --approach=find_unused --iterations=100

    # Full benchmark on GPU (4 GPUs, all approaches)
    NPROC=4 uv run python -m metta.rl.ddp_unused_params.benchmark \\
        --device=cuda --run-all --iterations=500

    # Single approach on GPU
    uv run torchrun --nproc-per-node=4 -m metta.rl.ddp_unused_params.benchmark \\
        --device=cuda --approach=hack --iterations=500

    # Compare specific approaches
    uv run torchrun --nproc-per-node=2 -m metta.rl.ddp_unused_params.benchmark \\
        --device=cuda --approach=find_unused --iterations=500
    uv run torchrun --nproc-per-node=2 -m metta.rl.ddp_unused_params.benchmark \\
        --device=cuda --approach=hack --iterations=500

Note: Use --run-all to automatically compare all approaches. For single approach testing,
use torchrun with --approach flag. The --run-all mode spawns separate subprocesses
for clean state between approaches.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


class RealisticPolicy(nn.Module):
    """A more realistic policy size to get meaningful benchmarks.

    Architecture similar to what's used in metta RL training.
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 512, num_actions: int = 64):
        super().__init__()
        # Encoder layers (like a small transformer or CNN)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor head (used by actor loss)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

        # Critic head (used by critic loss, unused in actor-only scenarios)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Auxiliary heads (sometimes unused)
        self.aux_head1 = nn.Linear(hidden_dim, 32)
        self.aux_head2 = nn.Linear(hidden_dim, 32)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encoder(x)
        logits = self.actor(hidden)
        values = self.critic(hidden)
        aux1 = self.aux_head1(hidden)
        aux2 = self.aux_head2(hidden)
        return {
            "logits": logits,
            "values": values,
            "aux1": aux1,
            "aux2": aux2,
            "hidden": hidden,
        }


@contextmanager
def cuda_timer(name: str, results: dict, device: torch.device):
    """Context manager for timing code blocks with proper GPU synchronization."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    yield
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    if name not in results:
        results[name] = []
    results[name].append(elapsed)


@contextmanager
def nullcontext():
    """Null context manager for warmup iterations."""
    yield


def run_benchmark(
    approach: str,
    device: torch.device,
    num_iterations: int = 500,
    batch_size: int = 256,
    warmup_iterations: int = 50,
    output_json: bool = False,
) -> dict[str, list[float]]:
    """Run benchmark for a specific approach."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # Set device for this process
    if device.type == "cuda":
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

    # Create model
    model = RealisticPolicy().to(device)
    param_count = sum(p.numel() for p in model.parameters())

    # Configure DDP based on approach
    if approach == "find_unused":
        # PyTorch's built-in solution
        find_unused = True
    elif approach == "hack":
        # We'll apply the hack manually, so DDP doesn't need to find unused
        find_unused = False
    else:  # all_used
        # All params are used, so no need for either
        find_unused = False

    ddp_kwargs = {"find_unused_parameters": find_unused}
    if device.type == "cuda":
        ddp_kwargs["device_ids"] = [local_rank]

    model = DDP(model, **ddp_kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    results: dict[str, list[float]] = {}

    if rank == 0 and not output_json:
        print(f"\n{'=' * 70}")
        print(f"Benchmarking: {approach}")
        print(f"  Device: {device}")
        print(f"  Parameters: {param_count:,}")
        print(f"  Batch size: {batch_size} x {world_size} processes = {batch_size * world_size} effective")
        print(f"  Iterations: {num_iterations} (warmup: {warmup_iterations})")
        print(f"  find_unused_parameters: {find_unused}")
        print(f"{'=' * 70}")

    # Extra GPU warmup
    if device.type == "cuda":
        for _ in range(10):
            x = torch.randn(batch_size, 512, device=device)
            outputs = model(x)
            # Use all outputs during warmup to avoid DDP issues
            loss = sum(v.sum() for v in outputs.values() if isinstance(v, torch.Tensor))
            loss.backward()
            optimizer.zero_grad()
        torch.cuda.synchronize(device)

    for i in range(num_iterations):
        is_warmup = i < warmup_iterations

        with cuda_timer("total", results, device) if not is_warmup else nullcontext():
            optimizer.zero_grad()

            # Forward pass
            x = torch.randn(batch_size, 512, device=device)

            with cuda_timer("forward", results, device) if not is_warmup else nullcontext():
                outputs = model(x)

            # Compute loss based on approach
            with cuda_timer("loss_compute", results, device) if not is_warmup else nullcontext():
                if approach == "all_used":
                    # Use ALL outputs in loss - baseline
                    loss = (
                        outputs["logits"].mean()
                        + outputs["values"].mean()
                        + outputs["aux1"].mean()
                        + outputs["aux2"].mean()
                    )
                elif approach == "hack":
                    # Only use logits, but add hack for unused params
                    loss = outputs["logits"].mean()
                    # The 0.0 * sum hack - add ALL other outputs
                    for key, value in outputs.items():
                        if key != "logits" and isinstance(value, torch.Tensor) and value.requires_grad:
                            loss = loss + 0.0 * value.sum()
                else:  # find_unused
                    # Only use logits, let DDP handle unused params
                    loss = outputs["logits"].mean()

            # Backward pass
            with cuda_timer("backward", results, device) if not is_warmup else nullcontext():
                loss.backward()

            # Optimizer step
            with cuda_timer("optimizer", results, device) if not is_warmup else nullcontext():
                optimizer.step()

    if dist.is_initialized():
        dist.barrier()

    # Compute and print statistics
    if rank == 0:
        if output_json:
            # Output JSON for programmatic parsing
            summary = {}
            for name, times in results.items():
                summary[name] = {
                    "mean_ms": sum(times) / len(times) * 1000,
                    "std_ms": (sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)) ** 0.5 * 1000,
                    "total_s": sum(times),
                }
            print(json.dumps({"approach": approach, "results": summary}))
        else:
            print(f"\nResults for {approach}:")
            print(f"  {'Phase':<15} {'Mean':>10} {'Std':>10} {'Total':>10}")
            print(f"  {'-' * 15} {'-' * 10} {'-' * 10} {'-' * 10}")
            for name, times in results.items():
                mean_ms = sum(times) / len(times) * 1000
                std_ms = (sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)) ** 0.5 * 1000
                total_s = sum(times)
                print(f"  {name:<15} {mean_ms:>8.3f}ms {std_ms:>8.3f}ms {total_s:>8.2f}s")

    return results


def run_all_approaches(args):
    """Run all approaches in separate subprocesses for clean state."""
    print("\n" + "=" * 70)
    print("RUNNING ALL APPROACHES (each in separate subprocess)")
    print("=" * 70)

    approaches = ["find_unused", "hack", "all_used"]
    all_results = {}

    for approach in approaches:
        print(f"\n>>> Running {approach}...")

        # Build torchrun command
        nproc = os.environ.get("NPROC", "2")
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc-per-node={nproc}",
            "--master-port=29510",
            "-m",
            "metta.rl.ddp_unused_params.benchmark",
            f"--device={args.device}",
            f"--iterations={args.iterations}",
            f"--warmup={args.warmup}",
            f"--batch-size={args.batch_size}",
            f"--approach={approach}",
            "--output-json",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                print(f"    ❌ FAILED: {approach}")
                print(f"    stderr: {result.stderr[:500]}")
                continue

            # Parse JSON output from last line
            for line in result.stdout.strip().split("\n"):
                if line.startswith("{"):
                    data = json.loads(line)
                    all_results[approach] = data["results"]
                    mean_ms = data["results"]["total"]["mean_ms"]
                    print(f"    ✅ {approach}: {mean_ms:.3f} ms/iter")
                    break
        except subprocess.TimeoutExpired:
            print(f"    ❌ TIMEOUT: {approach}")
        except Exception as e:
            print(f"    ❌ ERROR: {approach}: {e}")

    # Print comparison
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("SUMMARY COMPARISON")
        print("=" * 70)

        baseline_key = "find_unused" if "find_unused" in all_results else list(all_results.keys())[0]
        baseline_mean = all_results[baseline_key]["total"]["mean_ms"]

        print(f"\n{'Approach':<20} {'Mean (ms)':<12} {'Std (ms)':<12} {'vs ' + baseline_key:<20}")
        print("-" * 64)

        for approach, results in all_results.items():
            mean_ms = results["total"]["mean_ms"]
            std_ms = results["total"]["std_ms"]
            overhead_pct = (mean_ms / baseline_mean - 1) * 100
            overhead_str = f"{overhead_pct:+.1f}%" if approach != baseline_key else "baseline"
            print(f"{approach:<20} {mean_ms:<12.3f} {std_ms:<12.3f} {overhead_str:<20}")

        # Recommendation
        print("\n" + "=" * 70)
        print("RECOMMENDATION")
        print("=" * 70)
        if "find_unused" in all_results and "hack" in all_results:
            find_unused_mean = all_results["find_unused"]["total"]["mean_ms"]
            hack_mean = all_results["hack"]["total"]["mean_ms"]
            if find_unused_mean < hack_mean:
                faster = "find_unused_parameters=True"
                pct = (hack_mean / find_unused_mean - 1) * 100
            else:
                faster = "the 0.0*sum() hack"
                pct = (find_unused_mean / hack_mean - 1) * 100

            print(f"\nOn {args.device.upper()}: {faster} is {pct:.1f}% faster")
            print("\nFor TrainerConfig:")
            if "find_unused" in faster:
                print("  ddp_find_unused_parameters: true   # (default, recommended)")
            else:
                print("  ddp_find_unused_parameters: false  # (faster on this hardware)")


def main():
    """Run benchmarks with different configurations."""
    parser = argparse.ArgumentParser(
        description="Benchmark DDP unused parameter handling approaches. "
        "Compares find_unused_parameters=True vs the 0.0*sum() hack vs baseline."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=500,
        help="Number of training iterations to benchmark (default: 500)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Warmup iterations to skip before timing (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size per process (default: 256)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to benchmark on (default: cpu)",
    )
    parser.add_argument(
        "--approach",
        type=str,
        default=None,
        choices=["find_unused", "hack", "all_used"],
        help="Run only this approach. Use with torchrun. Options: find_unused (PyTorch built-in), "
        "hack (0.0*sum workaround), all_used (baseline with all params used)",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run all approaches in separate subprocesses and show comparison. "
        "Use this instead of torchrun for automatic comparison.",
    )
    parser.add_argument(
        "--output-json",
        action="store_true",
        help="Output results as JSON (used internally by --run-all)",
    )
    args = parser.parse_args()

    # If --run-all, spawn subprocesses for each approach
    if args.run_all:
        run_all_approaches(args)
        return

    # Single approach mode - must be run via torchrun
    if args.approach is None:
        print("Error: Must specify --approach or use --run-all")
        print("Example: torchrun --nproc-per-node=2 -m metta.rl.ddp_unused_params.benchmark --approach=find_unused")
        sys.exit(1)

    # Check GPU availability
    if args.device == "cuda":
        num_gpus = torch.cuda.device_count()
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if num_gpus < world_size:
            print(f"\n❌ ERROR: Requested {world_size} processes but only {num_gpus} GPU(s) available.")
            print(f"   Run with: --nproc-per-node={num_gpus}")
            return

    # Initialize distributed
    backend = "nccl" if args.device == "cuda" else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(args.device)

    if rank == 0 and not args.output_json:
        print("\n" + "=" * 70)
        print("DDP UNUSED PARAMETERS BENCHMARK")
        print("=" * 70)
        print(f"Backend: {backend}")
        print(f"Device: {args.device}")
        print(f"World size: {world_size}")
        if args.device == "cuda":
            num_gpus = torch.cuda.device_count()
            print(f"CUDA devices: {num_gpus}")
            for i in range(num_gpus):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            if world_size == 1:
                print("\n⚠️  Note: Running with 1 process - DDP sync overhead not measured")

    run_benchmark(
        approach=args.approach,
        device=device,
        num_iterations=args.iterations,
        warmup_iterations=args.warmup,
        batch_size=args.batch_size,
        output_json=args.output_json,
    )

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
