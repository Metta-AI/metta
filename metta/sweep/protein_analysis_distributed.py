#!/usr/bin/env python3
"""
Distributed GPU version of protein_analysis.py for multi-GPU acceleration.

This script runs optimization experiments across multiple GPUs, distributing
seeds across available devices for parallel execution.
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from tabulate import tabulate

# Add parent directory to path to import protein directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from protein import Protein  # Import directly to avoid sweep module conflicts

from metta.common.wandb.wandb_context import WandbConfig, WandbContext


@dataclass
class OptimizationStage:
    """Configuration for one stage of optimization."""

    name: str
    iterations: int
    ucb_beta: float = 2.0
    expansion_rate: float = 0.25
    suggestions_per_pareto: int = 64
    num_random_samples: int = 5


@dataclass
class ExperimentConfig:
    """Configuration for an optimization experiment."""

    problem_name: str
    acquisition_fn: str
    randomize_acquisition: bool
    stages: List[OptimizationStage]
    seeds: List[int]
    total_iterations: int = field(init=False)

    def __post_init__(self):
        self.total_iterations = sum(stage.iterations for stage in self.stages)


class OptimizationProblem:
    """Base class for optimization problems."""

    def __init__(self):
        self.name = "Base"
        self.dim = 2
        self.bounds = [(-1, 1)] * 2
        self.optimum = np.zeros(2)
        self.optimum_value = 0.0

    def evaluate(self, x: Dict[str, float]) -> float:
        raise NotImplementedError

    def get_protein_config(self) -> dict:
        """Get Protein configuration for this problem."""
        config = {
            "metric": self.name,
            "goal": "minimize",
            "method": "bayes",
        }

        # Add parameters with proper bounds
        for i in range(self.dim):
            param_name = f"x{i}"
            config[param_name] = {
                "min": self.bounds[i][0] if isinstance(self.bounds[i], tuple) else self.bounds[0],
                "max": self.bounds[i][1] if isinstance(self.bounds[i], tuple) else self.bounds[1],
                "distribution": "uniform",
                "mean": (
                    (self.bounds[i][0] + self.bounds[i][1]) / 2
                    if isinstance(self.bounds[i], tuple)
                    else (self.bounds[0] + self.bounds[1]) / 2
                ),
                "scale": "auto",
            }

        return config


class QuadraticProblem(OptimizationProblem):
    """Easy: 2D Quadratic function."""

    def __init__(self):
        super().__init__()
        self.name = "Quadratic2D"
        self.dim = 2
        self.bounds = [(-5, 5), (-5, 5)]
        self.optimum = np.array([2.0, -1.0])
        self.optimum_value = 0.0

    def evaluate(self, x: Dict[str, float]) -> float:
        x0, x1 = x["x0"], x["x1"]
        return (x0 - 2.0) ** 2 + (x1 + 1.0) ** 2


class BraninProblem(OptimizationProblem):
    """Medium: Branin function with 3 global minima."""

    def __init__(self):
        super().__init__()
        self.name = "Branin"
        self.dim = 2
        self.bounds = [(-5, 10), (0, 15)]
        self.optimum = np.array([np.pi, 2.275])
        self.optimum_value = 0.397887

    def evaluate(self, x: Dict[str, float]) -> float:
        x0, x1 = x["x0"], x["x1"]
        a = 1.0
        b = 5.1 / (4 * np.pi**2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)

        term1 = a * (x1 - b * x0**2 + c * x0 - r) ** 2
        term2 = s * (1 - t) * np.cos(x0)
        return term1 + term2 + s


class Hartmann6Problem(OptimizationProblem):
    """Hard: 6-dimensional Hartmann function."""

    def __init__(self):
        super().__init__()
        self.name = "Hartmann6D"
        self.dim = 6
        self.bounds = [(0, 1)] * 6
        self.optimum = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])
        self.optimum_value = -3.32237

        # Hartmann function parameters
        self.alpha = np.array([1.0, 1.2, 3.0, 3.2])
        self.A = np.array(
            [
                [10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14],
            ]
        )
        self.P = 1e-4 * np.array(
            [
                [1312, 1696, 5569, 124, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381],
            ]
        )

    def evaluate(self, x: Dict[str, float]) -> float:
        x_vec = np.array([x[f"x{i}"] for i in range(6)])

        outer_sum = 0
        for i in range(4):
            inner_sum = 0
            for j in range(6):
                inner_sum += self.A[i, j] * (x_vec[j] - self.P[i, j]) ** 2
            outer_sum += self.alpha[i] * np.exp(-inner_sum)

        return -outer_sum


def reset_torch_state(seed: int, device: str):
    """Reset PyTorch/Pyro state with a specific seed and device."""
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
    torch.manual_seed(seed)
    np.random.seed(seed)
    try:
        import pyro

        pyro.clear_param_store()
    except ImportError:
        pass


def run_single_optimization(
    problem: OptimizationProblem,
    config: ExperimentConfig,
    seed: int,
    device: str,
    rank: int,
    verbose: bool = True,
) -> Tuple[List[float], List[float], Dict[str, float]]:
    """Run a single optimization with multiple stages on a specific device.

    Returns:
        Tuple of (values_list, elapsed_times_list, best_params)
    """
    reset_torch_state(seed, device)

    if verbose and rank == 0:
        print(f"  [GPU {rank}] Seed {seed}: Starting optimization on {device}")

    all_values = []
    all_elapsed = []
    best_value = float("inf")
    best_params = None

    # Initialize optimizer with first stage config and device
    stage = config.stages[0]
    protein_config = problem.get_protein_config()

    optimizer = Protein(
        sweep_config=protein_config,
        acquisition_fn=config.acquisition_fn,
        num_random_samples=stage.num_random_samples,
        randomize_acquisition=config.randomize_acquisition,
        seed_with_search_center=False,
        ucb_beta=stage.ucb_beta,
        expansion_rate=stage.expansion_rate,
        suggestions_per_pareto=stage.suggestions_per_pareto,
        device=device,  # Pass device to Protein
    )

    iteration = 0
    for stage_idx, stage in enumerate(config.stages):
        if verbose and stage_idx > 0 and rank == 0:
            print(f"    [GPU {rank}] Switching to stage: {stage.name}")

        # Update optimizer parameters for new stage
        if stage_idx > 0:
            optimizer.ucb_beta = stage.ucb_beta
            optimizer.expansion_rate = stage.expansion_rate
            optimizer.suggestions_per_pareto = stage.suggestions_per_pareto

        for _ in range(stage.iterations):
            # Time this iteration
            iter_start = time.time()

            # Get suggestion
            suggestion, info = optimizer.suggest(fill=None)

            # Evaluate objective
            value = problem.evaluate(suggestion)
            all_values.append(value)

            # Track best
            if value < best_value:
                best_value = value
                best_params = suggestion.copy()

            # Provide feedback to optimizer
            optimizer.observe(suggestion, value, 1.0, is_failure=False)

            # Record elapsed time for this iteration
            iter_elapsed = time.time() - iter_start
            all_elapsed.append(iter_elapsed)

            iteration += 1
            if verbose and iteration % 10 == 0 and rank == 0:
                # Calculate iterations per second over the last 5 iterations
                window_start = max(0, iteration - 5)
                window_time = sum(all_elapsed[window_start:iteration])
                window_ips = 5 / window_time if window_time > 0 else 0
                print(
                    f"    [GPU {rank}] Seed {seed} Iter {iteration}: "
                    f"best = {best_value:.6f}\t ips: {window_ips:.2f} iter/s"
                )

    if verbose and rank == 0:
        total_elapsed = sum(all_elapsed)
        final_avg_ips = len(all_elapsed) / total_elapsed if total_elapsed > 0 else 0
        print(
            f"  [GPU {rank}] Seed {seed}: Final best = {best_value:.6f} "
            f"(time: {total_elapsed:.2f}s, avg: {final_avg_ips:.2f} iter/s)"
        )

    return all_values, all_elapsed, best_params


def _log_wandb_metrics(
    wandb_run,
    iteration_data: List[Dict],
    config: ExperimentConfig,
    problem: OptimizationProblem,
):
    """Log iteration-level metrics to WandB with seed aggregation."""
    if not iteration_data:
        return

    # Convert to DataFrame for easier aggregation
    iter_df = pd.DataFrame(iteration_data)

    # Group by iteration and compute statistics across seeds
    for iteration in range(1, config.total_iterations + 1):
        iter_data = iter_df[iter_df["iteration"] == iteration]

        if len(iter_data) == 0:
            continue

        # Compute aggregated metrics across seeds
        metrics = {
            "iteration": iteration,
            # Value statistics
            "value/median": iter_data["value"].median(),
            "value/mean": iter_data["value"].mean(),
            "value/std": iter_data["value"].std(),
            "value/min": iter_data["value"].min(),
            "value/max": iter_data["value"].max(),
            # Best so far statistics
            "best/median": iter_data["best_so_far"].median(),
            "best/mean": iter_data["best_so_far"].mean(),
            "best/std": iter_data["best_so_far"].std(),
            "best/min": iter_data["best_so_far"].min(),
            "best/max": iter_data["best_so_far"].max(),
            # Time statistics
            "time/median": iter_data["elapsed"].median(),
            "time/mean": iter_data["elapsed"].mean(),
            # Distance to optimum
            "error/median": abs(iter_data["best_so_far"].median() - problem.optimum_value),
            "error/min": abs(iter_data["best_so_far"].min() - problem.optimum_value),
        }

        # Add per-seed values for detailed tracking
        for seed in config.seeds:
            seed_data = iter_data[iter_data["seed"] == seed]
            if len(seed_data) > 0:
                metrics[f"seed_{seed}/value"] = seed_data["value"].iloc[0]
                metrics[f"seed_{seed}/best"] = seed_data["best_so_far"].iloc[0]

        wandb_run.log(metrics, step=iteration)


def _find_convergence_iteration(df: pd.DataFrame, optimum_value: float, tolerance: float = 0.01) -> int:
    """Find the iteration where the median best value converges to within tolerance of optimum."""
    if "best_median" not in df:
        return -1

    for i, val in enumerate(df["best_median"]):
        if abs(val - optimum_value) <= tolerance:
            return i + 1

    return -1


def _process_results(
    results: Dict,
    elapsed_results: Dict,
    config: ExperimentConfig,
    problem: OptimizationProblem,
    output_dir: Path,
    verbose: bool,
) -> pd.DataFrame:
    """Process results and create DataFrame."""
    # Create DataFrame with results
    df = pd.DataFrame(results)
    df["iteration"] = range(1, len(df) + 1)

    # Add elapsed time columns for each seed
    for seed in config.seeds:
        if f"elapsed_seed_{seed}" in elapsed_results:
            df[f"elapsed_seed_{seed}"] = elapsed_results[f"elapsed_seed_{seed}"]

    # Add elapsed time statistics
    elapsed_cols = [f"elapsed_seed_{s}" for s in config.seeds if f"elapsed_seed_{s}" in df.columns]
    if elapsed_cols:
        df["elapsed_median"] = df[elapsed_cols].median(axis=1)
        df["elapsed_mean"] = df[elapsed_cols].mean(axis=1)
        df["elapsed_std"] = df[elapsed_cols].std(axis=1)
        df["elapsed_min"] = df[elapsed_cols].min(axis=1)
        df["elapsed_max"] = df[elapsed_cols].max(axis=1)

    # Add median and statistics for values
    value_cols = [f"seed_{s}" for s in config.seeds if f"seed_{s}" in df.columns]
    df["median"] = df[value_cols].median(axis=1)
    df["mean"] = df[value_cols].mean(axis=1)
    df["std"] = df[value_cols].std(axis=1)
    df["min"] = df[value_cols].min(axis=1)
    df["max"] = df[value_cols].max(axis=1)

    # Add cumulative best for each seed and median
    for seed in config.seeds:
        if f"seed_{seed}" in df.columns:
            df[f"best_seed_{seed}"] = df[f"seed_{seed}"].cummin()
    df["best_median"] = df["median"].cummin()

    # Save results
    filename = f"{problem.name}_{config.acquisition_fn}_randomize_{config.randomize_acquisition}.csv"
    filepath = output_dir / filename
    df.to_csv(filepath, index=False)

    if verbose:
        print(f"\n--Results saved to: {filepath}")
        print(f"+Final median best: {df['best_median'].iloc[-1]:.6f}")
        print(f"+Target optimum: {problem.optimum_value:.6f}")
        print(f"+Error: {abs(df['best_median'].iloc[-1] - problem.optimum_value):.6f}")
        if "elapsed_median" in df.columns:
            print(
                f"+Total median time: {df['elapsed_median'].sum():.2f}s (mean: {df['elapsed_mean'].mean():.4f}s/iter)"
            )

    return df


def worker_process(
    rank: int,
    world_size: int,
    problem: OptimizationProblem,
    config: ExperimentConfig,
    output_dir: Path,
    verbose: bool,
    wandb_config: Optional[WandbConfig] = None,
):
    """Worker process that runs on a single GPU."""
    # Set up device
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"

    # Distribute seeds across workers
    seeds_per_worker = len(config.seeds) // world_size
    remainder = len(config.seeds) % world_size

    # Calculate seed range for this worker
    if rank < remainder:
        start_idx = rank * (seeds_per_worker + 1)
        end_idx = start_idx + seeds_per_worker + 1
    else:
        start_idx = rank * seeds_per_worker + remainder
        end_idx = start_idx + seeds_per_worker

    my_seeds = config.seeds[start_idx:end_idx]

    if verbose and rank == 0:
        print(f"\n{'=' * 80}")
        print(f"Distributed execution on {world_size} GPUs")
        print(f"Problem: {problem.name}")
        print(f"Acquisition: {config.acquisition_fn} (randomize={config.randomize_acquisition})")
        print(f"Total seeds: {len(config.seeds)}")
        print(f"Seeds per GPU: ~{seeds_per_worker}")
        print(f"{'=' * 80}")

    # Run optimization for assigned seeds
    results = {}
    elapsed_results = {}

    # Collect iteration-level data for WandB logging
    all_iteration_data = []

    for seed in my_seeds:
        values, elapsed_times, best_params = run_single_optimization(problem, config, seed, device, rank, verbose)
        results[f"seed_{seed}"] = values
        elapsed_results[f"elapsed_seed_{seed}"] = elapsed_times

        # Store per-iteration data for this seed
        for i, (val, elapsed) in enumerate(zip(values, elapsed_times, strict=False)):
            all_iteration_data.append(
                {
                    "iteration": i + 1,
                    "seed": seed,
                    "value": val,
                    "elapsed": elapsed,
                    "best_so_far": min(values[: i + 1]),
                }
            )

    # Gather results from all workers
    if dist.is_initialized():
        # Gather all results to rank 0
        all_results = [None] * world_size
        all_elapsed = [None] * world_size
        all_iter_data = [None] * world_size
        dist.gather_object(results, all_results if rank == 0 else None, dst=0)
        dist.gather_object(elapsed_results, all_elapsed if rank == 0 else None, dst=0)
        dist.gather_object(all_iteration_data, all_iter_data if rank == 0 else None, dst=0)

        if rank == 0:
            # Combine results from all workers
            combined_results = {}
            combined_elapsed = {}
            combined_iter_data = []
            for worker_results, worker_elapsed, worker_iter_data in zip(
                all_results, all_elapsed, all_iter_data, strict=False
            ):
                combined_results.update(worker_results)
                combined_elapsed.update(worker_elapsed)
                combined_iter_data.extend(worker_iter_data)
            results = combined_results
            elapsed_results = combined_elapsed
            all_iteration_data = combined_iter_data
    else:
        # Single process mode - results are already complete
        pass

    # Only rank 0 saves results, logs to wandb, and prints summary
    if rank == 0 or not dist.is_initialized():
        # Initialize WandB if configured
        wandb_run = None
        if wandb_config:
            # Create descriptive run name
            run_name = (
                f"protein_opt.analysis.{problem.name.lower()}_"
                f"{config.acquisition_fn}_"
                f"{'rand' if config.randomize_acquisition else 'det'}_"
                f"{len(config.seeds)}seeds_"
                f"{config.total_iterations}iter"
            )

            wandb_config = WandbConfig(
                enabled=True,
                project=wandb_config.project if hasattr(wandb_config, "project") else "metta",
                entity=wandb_config.entity if hasattr(wandb_config, "entity") else "",
                name=run_name,
                tags=["sweep", "protein-opt-diagnostics"],
                group="protein-opt-diagnostics",
            )

            with WandbContext(wandb_config) as wandb_run:
                # Log configuration
                wandb_run.config.update(
                    {
                        "problem": problem.name,
                        "problem_dim": problem.dim,
                        "problem_optimum": problem.optimum_value,
                        "acquisition_fn": config.acquisition_fn,
                        "randomize_acquisition": config.randomize_acquisition,
                        "num_seeds": len(config.seeds),
                        "total_iterations": config.total_iterations,
                        "num_gpus": world_size,
                        "stages": [
                            {
                                "name": s.name,
                                "iterations": s.iterations,
                                "ucb_beta": s.ucb_beta,
                                "expansion_rate": s.expansion_rate,
                            }
                            for s in config.stages
                        ],
                    }
                )

                # Process and log iteration-level metrics
                _log_wandb_metrics(wandb_run, all_iteration_data, config, problem)

                # Continue with regular processing
                df = _process_results(results, elapsed_results, config, problem, output_dir, verbose)

                # Log final summary metrics
                wandb_run.summary.update(
                    {
                        "final_best_median": df["best_median"].iloc[-1],
                        "final_error": abs(df["best_median"].iloc[-1] - problem.optimum_value),
                        "total_time_median": df["elapsed_median"].sum() if "elapsed_median" in df else 0,
                        "convergence_iteration": _find_convergence_iteration(df, problem.optimum_value),
                    }
                )
        else:
            # No wandb - just process results normally
            df = _process_results(results, elapsed_results, config, problem, output_dir, verbose)

        return df
    return None


def run_distributed(
    problem: OptimizationProblem,
    config: ExperimentConfig,
    output_dir: Path,
    verbose: bool = True,
    wandb_config: Optional[WandbConfig] = None,
) -> Optional[pd.DataFrame]:
    """Run distributed optimization if torch.distributed is available."""
    if "LOCAL_RANK" in os.environ:
        # Running in distributed mode
        _ = int(os.environ["LOCAL_RANK"])  # Validate LOCAL_RANK exists
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        # Running in single-process mode
        rank = 0
        world_size = 1

    # Run worker process
    df = worker_process(rank, world_size, problem, config, output_dir, verbose, wandb_config)

    # Clean up distributed process group
    if dist.is_initialized():
        dist.destroy_process_group()

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Distributed GPU analysis of Protein optimizer performance",
        epilog="""
Examples:
  # Single-GPU test
  %(prog)s --nseeds=8 --acquisition=ucb --problem=quadratic --verbose
  
  # Multi-GPU execution (requires torchrun)
  torchrun --nproc_per_node=8 %(prog)s --nseeds=32 --acquisition=all --verbose
  
  # Using run_protein_distributed.sh script
  ./run_protein_distributed.sh --nseeds=64 --stages=adaptive

This distributed version automatically distributes seeds across available GPUs
for parallel execution, significantly speeding up multi-seed experiments.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--nseeds",
        type=int,
        default=8,
        help="Number of random seeds to run for each configuration (default: 8)",
    )
    parser.add_argument(
        "--acquisition",
        type=str,
        default="all",
        choices=["all", "naive", "ei", "ucb"],
        help="Acquisition function to test. 'all' runs all three (default: all)",
    )
    parser.add_argument(
        "--randomize-acquisition",
        type=str,
        default="both",
        choices=["both", "true", "false"],
        help="Whether to randomize acquisition parameters (default: both)",
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="all",
        choices=["all", "quadratic", "branin", "hartmann"],
        help="Optimization problem to solve. 'all' runs all three (default: all)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="./protein_analysis_distributed/",
        help="Output directory for results (default: ./protein_analysis_distributed/)",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="standard",
        choices=["standard", "adaptive", "custom"],
        help="Optimization stage configuration (default: standard)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress during optimization",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable WandB logging with seed aggregation",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="WandB entity (team/user name)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="metta",
        help="WandB project name (default: metta)",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate seeds
    seeds = list(range(42, 42 + args.nseeds))

    # Define stages based on configuration
    if args.stages == "standard":
        stages = [OptimizationStage(name="balanced", iterations=70, ucb_beta=2.0, expansion_rate=0.25)]
    elif args.stages == "adaptive":
        stages = [
            OptimizationStage(
                name="explore", iterations=30, ucb_beta=3.0, expansion_rate=0.5, suggestions_per_pareto=128
            ),
            OptimizationStage(
                name="balance", iterations=20, ucb_beta=2.0, expansion_rate=0.25, suggestions_per_pareto=64
            ),
            OptimizationStage(
                name="exploit", iterations=20, ucb_beta=1.0, expansion_rate=0.1, suggestions_per_pareto=32
            ),
        ]
    else:  # custom
        stages = [OptimizationStage(name="full", iterations=50)]

    # Set up wandb config if enabled
    wandb_config = None
    if args.wandb:
        wandb_config = WandbConfig(
            enabled=True,
            project=args.wandb_project,
            entity=args.wandb_entity if args.wandb_entity else "",
        )

    # Select problems
    all_problems = {
        "quadratic": QuadraticProblem(),
        "branin": BraninProblem(),
        "hartmann": Hartmann6Problem(),
    }

    if args.problem == "all":
        problems = all_problems.values()
    else:
        problems = [all_problems[args.problem]]

    # Select acquisition functions
    if args.acquisition == "all":
        acquisition_fns = ["naive", "ei", "ucb"]
    else:
        acquisition_fns = [args.acquisition]

    # Select randomization settings
    if args.randomize_acquisition == "both":
        randomize_settings = [False, True]
    elif args.randomize_acquisition == "true":
        randomize_settings = [True]
    else:
        randomize_settings = [False]

    # Run experiments
    start_time = time.time()

    for problem in problems:
        problem_dir = output_dir / problem.name
        problem_dir.mkdir(exist_ok=True)

        all_results = {}

        for acq_fn in acquisition_fns:
            for randomize in randomize_settings:
                config = ExperimentConfig(
                    problem_name=problem.name,
                    acquisition_fn=acq_fn,
                    randomize_acquisition=randomize,
                    stages=stages,
                    seeds=seeds,
                )

                df = run_distributed(problem, config, problem_dir, verbose=args.verbose, wandb_config=wandb_config)

                if df is not None:  # Only rank 0 gets the DataFrame
                    result_key = f"{problem.name}_{acq_fn}_randomize_{randomize}"
                    all_results[result_key] = df

        # Create and save summary (only on rank 0 or single process)
        if all_results:
            try:
                from protein_analysis import create_summary_table
            except ImportError:
                # If protein_analysis not available, create a simple summary
                def create_summary_table(results, problem):
                    import pandas as pd

                    summary_data = []
                    for config_name, df in results.items():
                        summary_data.append(
                            {
                                "Config": config_name,
                                "Final Best": df["best_median"].iloc[-1] if "best_median" in df else 0,
                                "Error": abs(df["best_median"].iloc[-1] - problem.optimum_value)
                                if "best_median" in df
                                else 0,
                            }
                        )
                    return pd.DataFrame(summary_data)

            summary_df = create_summary_table(all_results, problem)
            summary_path = problem_dir / "summary.csv"
            summary_df.to_csv(summary_path, index=False)

            print(f"\n{'=' * 80}")
            print(f"Summary for {problem.name}")
            print(f"{'=' * 80}")
            print(tabulate(summary_df, headers="keys", tablefmt="grid", floatfmt=".6f"))
            print(f"Summary saved to: {summary_path}")

    if not dist.is_initialized() or dist.get_rank() == 0:
        elapsed = time.time() - start_time
        print(f"\n{'=' * 80}")
        print(f"All experiments completed in {elapsed:.1f} seconds")
        print(f"Results saved to: {output_dir}")
        print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
