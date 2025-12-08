#!/usr/bin/env python3
"""Benchmark DDP unused parameter handling approaches.

Compares:
1. find_unused_parameters=True (PyTorch built-in)
2. The 0.0 * value.sum() hack
3. Baseline with all params used (no unused params)

## Running on CPU (2 processes):
    uv run torchrun --nproc-per-node=2 --master-port=29510 scripts/benchmark_ddp_approaches.py

## Running on GPU (uses NCCL, requires 2+ GPUs):
    uv run torchrun --nproc-per-node=2 --master-port=29510 scripts/benchmark_ddp_approaches.py --device=cuda

## Running on GPU cluster (e.g., 4 GPUs):
    uv run torchrun --nproc-per-node=4 --master-port=29510 scripts/benchmark_ddp_approaches.py --device=cuda --iterations=1000

## Quick single-approach test:
    uv run torchrun --nproc-per-node=2 scripts/benchmark_ddp_approaches.py --device=cuda --approach=find_unused

## Compare all approaches with summary:
    uv run torchrun --nproc-per-node=2 scripts/benchmark_ddp_approaches.py --device=cuda --all
"""

import argparse
import os
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
) -> dict[str, list[float]]:
    """Run benchmark for a specific approach.

    Args:
        approach: One of "find_unused", "hack", "all_used"
        device: Device to run on (cpu or cuda)
        num_iterations: Number of training iterations
        batch_size: Batch size per process
        warmup_iterations: Iterations to skip for timing

    Returns:
        Dictionary of timing results
    """
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

    # Wrap in DDP
    if approach == "find_unused":
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None, find_unused_parameters=True)
    else:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None, find_unused_parameters=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    results: dict[str, list[float]] = {}

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"Benchmarking: {approach}")
        print(f"  Device: {device}")
        print(f"  Parameters: {param_count:,}")
        print(f"  Batch size: {batch_size} x {world_size} processes = {batch_size * world_size} effective")
        print(f"  Iterations: {num_iterations} (warmup: {warmup_iterations})")
        print(f"{'='*70}")

    # Extra GPU warmup
    if device.type == "cuda":
        for _ in range(10):
            x = torch.randn(batch_size, 512, device=device)
            outputs = model(x)
            loss = outputs["logits"].mean()
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

            # Compute loss (only using logits, not values/aux)
            with cuda_timer("loss_compute", results, device) if not is_warmup else nullcontext():
                if approach == "all_used":
                    # Use ALL outputs in loss
                    loss = (
                        outputs["logits"].mean()
                        + outputs["values"].mean()
                        + outputs["aux1"].mean()
                        + outputs["aux2"].mean()
                    )
                else:
                    # Only use logits (like ppo_actor)
                    loss = outputs["logits"].mean()

                    if approach == "hack":
                        # The 0.0 * sum hack
                        for key, value in outputs.items():
                            if key != "logits" and isinstance(value, torch.Tensor):
                                if value.requires_grad:
                                    loss = loss + 0.0 * value.sum()

            # Backward pass
            with cuda_timer("backward", results, device) if not is_warmup else nullcontext():
                loss.backward()

            # Optimizer step
            with cuda_timer("optimizer", results, device) if not is_warmup else nullcontext():
                optimizer.step()

    dist.barrier()

    # Compute and print statistics
    if rank == 0:
        print(f"\nResults for {approach}:")
        print(f"  {'Phase':<15} {'Mean':>10} {'Std':>10} {'Total':>10}")
        print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10}")
        for name, times in results.items():
            mean_ms = sum(times) / len(times) * 1000
            std_ms = (sum((t - mean_ms / 1000) ** 2 for t in times) / len(times)) ** 0.5 * 1000
            total_s = sum(times)
            print(f"  {name:<15} {mean_ms:>8.3f}ms {std_ms:>8.3f}ms {total_s:>8.2f}s")

    return results


def main():
    """Run benchmarks with different configurations."""
    parser = argparse.ArgumentParser(description="Benchmark DDP unused parameter handling")
    parser.add_argument("--iterations", type=int, default=500, help="Number of iterations per approach")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations (not timed)")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size per process")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to benchmark on")
    parser.add_argument(
        "--approach",
        type=str,
        default=None,
        choices=["find_unused", "hack", "all_used"],
        help="Run only this approach (default: run all)",
    )
    parser.add_argument("--all", action="store_true", help="Run all approaches and show comparison summary")
    args = parser.parse_args()

    # Initialize distributed
    backend = "nccl" if args.device == "cuda" else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    device = torch.device(args.device)

    if rank == 0:
        print("\n" + "=" * 70)
        print("DDP UNUSED PARAMETERS BENCHMARK")
        print("=" * 70)
        print(f"Backend: {backend}")
        print(f"Device: {args.device}")
        print(f"World size: {dist.get_world_size()}")
        if args.device == "cuda":
            print(f"CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Determine which approaches to run
    if args.approach:
        approaches = [args.approach]
    elif args.all:
        approaches = ["find_unused", "hack", "all_used"]
    else:
        approaches = ["find_unused", "hack", "all_used"]

    all_results: dict[str, dict[str, list[float]]] = {}

    for approach in approaches:
        # Re-init distributed for each approach to get clean state
        if approach != approaches[0]:
            dist.barrier()

        results = run_benchmark(
            approach=approach,
            device=device,
            num_iterations=args.iterations,
            warmup_iterations=args.warmup,
            batch_size=args.batch_size,
        )
        all_results[approach] = results

    # Print comparison summary
    if rank == 0 and len(all_results) > 1:
        print("\n" + "=" * 70)
        print("SUMMARY COMPARISON")
        print("=" * 70)

        # Use find_unused as baseline if available, otherwise first approach
        baseline_key = "find_unused" if "find_unused" in all_results else list(all_results.keys())[0]
        baseline_times = all_results[baseline_key].get("total", [])
        baseline_mean = sum(baseline_times) / len(baseline_times) if baseline_times else 1

        print(f"\n{'Approach':<20} {'Mean (ms)':<12} {'Std (ms)':<12} {'vs ' + baseline_key:<20}")
        print("-" * 64)

        for approach, results in all_results.items():
            total_times = results.get("total", [])
            if total_times:
                mean_ms = sum(total_times) / len(total_times) * 1000
                std_ms = (sum((t - mean_ms / 1000) ** 2 for t in total_times) / len(total_times)) ** 0.5 * 1000
                mean_s = sum(total_times) / len(total_times)
                overhead_pct = (mean_s / baseline_mean - 1) * 100
                overhead_str = f"{overhead_pct:+.1f}%" if approach != baseline_key else "baseline"
                print(f"{approach:<20} {mean_ms:<12.3f} {std_ms:<12.3f} {overhead_str:<20}")

        # Statistical significance note
        print("\n" + "-" * 64)
        n = len(all_results[baseline_key].get("total", []))
        print(f"Note: {n} iterations measured. Standard error ≈ std/√{n}")
        print("Results are statistically significant if confidence intervals don't overlap.")

        # Recommendation
        print("\n" + "=" * 70)
        print("RECOMMENDATION")
        print("=" * 70)
        if "find_unused" in all_results and "hack" in all_results:
            find_unused_mean = sum(all_results["find_unused"]["total"]) / len(all_results["find_unused"]["total"])
            hack_mean = sum(all_results["hack"]["total"]) / len(all_results["hack"]["total"])
            if find_unused_mean < hack_mean:
                faster = "find_unused_parameters=True"
                slower = "the 0.0*sum() hack"
                pct = (hack_mean / find_unused_mean - 1) * 100
            else:
                faster = "the 0.0*sum() hack"
                slower = "find_unused_parameters=True"
                pct = (find_unused_mean / hack_mean - 1) * 100

            print(f"\nOn {args.device.upper()}: {faster} is {pct:.1f}% faster than {slower}")
            print("\nFor TrainerConfig:")
            if "find_unused" in faster:
                print("  ddp_find_unused_parameters: true   # (default, recommended)")
            else:
                print("  ddp_find_unused_parameters: false  # (faster on this hardware)")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
