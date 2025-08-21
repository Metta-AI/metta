#!/usr/bin/env python3
"""
Comprehensive analysis of Protein optimizer performance with multiple seeds and configurations.

This script evaluates the Protein optimizer on canonical optimization problems with:
- Multiple random seeds for statistical robustness
- Different acquisition functions and configurations
- Multi-stage optimization strategies
- Detailed per-iteration tracking
- CSV output for analysis
"""

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tabulate import tabulate

from metta.sweep.protein import Protein


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


def reset_torch_state(seed: int):
    """Reset PyTorch/Pyro state with a specific seed."""
    if torch.cuda.is_available():
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
    verbose: bool = True,
) -> Tuple[List[float], List[float], Dict[str, float]]:
    """Run a single optimization with multiple stages.

    Returns:
        Tuple of (values_list, elapsed_times_list, best_params)
    """
    reset_torch_state(seed)

    if verbose:
        print(f"  --Seed {seed}: Starting optimization")

    all_values = []
    all_elapsed = []
    best_value = float("inf")
    best_params = None

    # Initialize optimizer with first stage config
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
    )

    iteration = 0
    for stage_idx, stage in enumerate(config.stages):
        if verbose and stage_idx > 0:
            print(f"    \n# Switching to stage: {stage.name}\n")

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
            if verbose and iteration % 5 == 0:
                # Calculate iterations per second over the last 5 iterations
                window_start = max(0, iteration - 5)
                window_time = sum(all_elapsed[window_start:iteration])
                window_ips = 5 / window_time if window_time > 0 else 0
                print(f"    +Iteration {iteration}: best = {best_value:.6f}\t avg_ips: {window_ips:.2f} iter/s")

    if verbose:
        total_elapsed = sum(all_elapsed)
        final_avg_ips = len(all_elapsed) / total_elapsed if total_elapsed > 0 else 0
        print(
            f"  +Seed {seed}: Final best = {best_value:.6f}"
            f"(total time: {total_elapsed:.2f}s, avg: {final_avg_ips:.2f} iter/s)"
        )

    return all_values, all_elapsed, best_params


def run_experiment(
    problem: OptimizationProblem,
    config: ExperimentConfig,
    output_dir: Path,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run a complete experiment with multiple seeds."""
    if verbose:
        print(f"\n{'=' * 80}")
        print(f"Running {problem.name} with {config.acquisition_fn} (randomize={config.randomize_acquisition})")
        print(f"Stages: {', '.join(f'{s.name}({s.iterations})' for s in config.stages)}")
        print(f"Seeds: {len(config.seeds)}")
        print(f"{'=' * 80}")

    # Run optimization for each seed
    results = {}
    elapsed_results = {}
    for seed in config.seeds:
        values, elapsed_times, best_params = run_single_optimization(problem, config, seed, verbose)
        results[f"seed_{seed}"] = values
        elapsed_results[f"elapsed_seed_{seed}"] = elapsed_times

    # Create DataFrame with results
    df = pd.DataFrame(results)
    df["iteration"] = range(1, len(df) + 1)

    # Add elapsed time columns for each seed
    for seed in config.seeds:
        df[f"elapsed_seed_{seed}"] = elapsed_results[f"elapsed_seed_{seed}"]

    # Add elapsed time statistics
    elapsed_cols = [f"elapsed_seed_{s}" for s in config.seeds]
    df["elapsed_median"] = df[elapsed_cols].median(axis=1)
    df["elapsed_mean"] = df[elapsed_cols].mean(axis=1)
    df["elapsed_std"] = df[elapsed_cols].std(axis=1)
    df["elapsed_min"] = df[elapsed_cols].min(axis=1)
    df["elapsed_max"] = df[elapsed_cols].max(axis=1)

    # Add median and statistics for values
    value_cols = [f"seed_{s}" for s in config.seeds]
    df["median"] = df[value_cols].median(axis=1)
    df["mean"] = df[value_cols].mean(axis=1)
    df["std"] = df[value_cols].std(axis=1)
    df["min"] = df[value_cols].min(axis=1)
    df["max"] = df[value_cols].max(axis=1)

    # Add cumulative best for each seed and median
    for seed in config.seeds:
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
        print(f"+Total median time: {df['elapsed_median'].sum():.2f}s (mean: {df['elapsed_mean'].mean():.4f}s/iter)")

    return df


def create_summary_table(results: Dict[str, pd.DataFrame], problem: OptimizationProblem) -> pd.DataFrame:
    """Create a summary table of all experiments."""
    summary_data = []

    for config_name, df in results.items():
        final_median = df["best_median"].iloc[-1]
        final_std = df[[col for col in df.columns if col.startswith("best_seed_")]].iloc[-1].std()
        error = abs(final_median - problem.optimum_value)

        # Calculate timing statistics
        total_time = df["elapsed_median"].sum()
        mean_time_per_iter = df["elapsed_mean"].mean()

        # Parse config name
        parts = config_name.split("_")
        acq_fn = parts[1]
        randomize = parts[-1] == "True"

        summary_data.append(
            {
                "Acquisition": acq_fn,
                "Randomize": randomize,
                "Final Median": final_median,
                "Final Std": final_std,
                "Error": error,
                "Total Time (s)": total_time,
                "Mean Time/Iter": mean_time_per_iter,
                "Iter 10": df["best_median"].iloc[9] if len(df) > 9 else None,
                "Iter 30": df["best_median"].iloc[29] if len(df) > 29 else None,
                "Iter 50": df["best_median"].iloc[49] if len(df) > 49 else None,
            }
        )

    return pd.DataFrame(summary_data)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Protein optimizer performance with multiple seeds and configurations",
        epilog="""
Examples:
  # Quick test with UCB on quadratic problem
  %(prog)s --nseeds=3 --acquisition=ucb --problem=quadratic --verbose
  
  # Full analysis with adaptive stages on all problems
  %(prog)s --nseeds=10 --stages=adaptive --verbose
  
  # Compare all acquisition functions on Hartmann problem
  %(prog)s --nseeds=5 --problem=hartmann --acquisition=all
  
Stage Configurations:
  standard: Single balanced stage with 70 iterations
  adaptive: Three stages - explore(30) -> balance(20) -> exploit(20)
  custom:   Single stage with 50 iterations (faster for testing)

Output:
  Creates CSV files with per-iteration results for each configuration.
  Saves summary statistics comparing all tested configurations.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--nseeds",
        type=int,
        default=5,
        help="Number of random seeds to run for each configuration (default: 5)",
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
        help="Whether to randomize acquisition function parameters. 'both' tests with and without (default: both)",
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
        default="./protein_analysis/",
        help="Output directory for results CSV files (default: ./protein_analysis/)",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="standard",
        choices=["standard", "adaptive", "custom"],
        help="Optimization stage configuration - see above for details (default: standard)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress during optimization (default: False)",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate seeds
    seeds = list(range(42, 42 + args.nseeds))

    # Define stages based on configuration
    if args.stages == "standard":
        # Single stage with balanced parameters
        stages = [OptimizationStage(name="balanced", iterations=70, ucb_beta=2.0, expansion_rate=0.25)]
    elif args.stages == "adaptive":
        # Three-stage optimization: explore -> balance -> exploit
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
        # You can define custom stages here
        stages = [OptimizationStage(name="full", iterations=50)]

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

                df = run_experiment(problem, config, problem_dir, verbose=args.verbose)
                result_key = f"{problem.name}_{acq_fn}_randomize_{randomize}"
                all_results[result_key] = df

        # Create and save summary
        if all_results:
            summary_df = create_summary_table(all_results, problem)
            summary_path = problem_dir / "summary.csv"
            summary_df.to_csv(summary_path, index=False)

            print(f"\n{'=' * 80}")
            print(f"Summary for {problem.name}")
            print(f"{'=' * 80}")
            print(tabulate(summary_df, headers="keys", tablefmt="grid", floatfmt=".6f"))
            print(f"Summary saved to: {summary_path}")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 80}")
    print(f"All experiments completed in {elapsed:.1f} seconds")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
