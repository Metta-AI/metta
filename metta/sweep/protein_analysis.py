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
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tabulate import tabulate

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

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


def _optimization_worker(args: Tuple) -> Tuple[int, List[float], List[float], Dict[str, float]]:
    """Worker function for parallel optimization. Returns seed and results."""
    problem, config, seed, verbose = args
    values, elapsed, best_params = run_single_optimization(problem, config, seed, verbose)
    return seed, values, elapsed, best_params


def run_experiment(
    problem: OptimizationProblem,
    config: ExperimentConfig,
    output_dir: Path,
    verbose: bool = True,
    wandb_config: Optional[Dict[str, Any]] = None,
    n_workers: Optional[int] = None,
) -> pd.DataFrame:
    """Run a complete experiment with multiple seeds using parallel processing.

    Args:
        problem: Optimization problem to solve
        config: Experiment configuration
        output_dir: Directory to save results
        verbose: Whether to print progress
        wandb_config: Optional WandB configuration
        n_workers: Number of parallel workers (default: cpu_count)
    """
    if verbose:
        print(f"\n{'=' * 80}")
        print(f"Running {problem.name} with {config.acquisition_fn} (randomize={config.randomize_acquisition})")
        print(f"Stages: {', '.join(f'{s.name}({s.iterations})' for s in config.stages)}")
        print(f"Seeds: {len(config.seeds)}")

    # Determine number of workers
    if n_workers is None:
        n_workers = min(cpu_count(), len(config.seeds))

    # Ensure num_seeds % num_workers == 0
    if len(config.seeds) % n_workers != 0:
        # Adjust number of workers to be a divisor of num_seeds
        for w in range(n_workers, 0, -1):
            if len(config.seeds) % w == 0:
                n_workers = w
                break

    if verbose:
        print(f"Parallel workers: {n_workers}")
        print(f"{'=' * 80}")

    # Collect iteration-level data for WandB logging
    all_iteration_data = []

    # Prepare arguments for parallel workers
    worker_args = [(problem, config, seed, verbose) for seed in config.seeds]

    # Run optimization in parallel
    results = {}
    elapsed_results = {}

    if n_workers > 1:
        # Use multiprocessing
        with Pool(n_workers) as pool:
            worker_results = pool.map(_optimization_worker, worker_args)

        # Process results
        for seed, values, elapsed_times, best_params in worker_results:
            results[f"seed_{seed}"] = values
            elapsed_results[f"elapsed_seed_{seed}"] = elapsed_times

            # Store per-iteration data for this seed
            for i, (val, elapsed) in enumerate(zip(values, elapsed_times)):
                best_so_far = min(values[: i + 1])
                all_iteration_data.append(
                    {
                        "iteration": i + 1,
                        "seed": seed,
                        "value": val,
                        "elapsed": elapsed,
                        "best_so_far": best_so_far,
                        "error": abs(val - problem.optimum_value),  # Error of current sample
                        "best_error": abs(best_so_far - problem.optimum_value),  # Best error achieved so far
                    }
                )
    else:
        # Sequential execution (for debugging or when n_workers=1)
        for seed in config.seeds:
            values, elapsed_times, best_params = run_single_optimization(problem, config, seed, verbose)
            results[f"seed_{seed}"] = values
            elapsed_results[f"elapsed_seed_{seed}"] = elapsed_times

            # Store per-iteration data for this seed
            for i, (val, elapsed) in enumerate(zip(values, elapsed_times)):
                best_so_far = min(values[: i + 1])
                all_iteration_data.append(
                    {
                        "iteration": i + 1,
                        "seed": seed,
                        "value": val,
                        "elapsed": elapsed,
                        "best_so_far": best_so_far,
                        "error": abs(val - problem.optimum_value),  # Error of current sample
                        "best_error": abs(best_so_far - problem.optimum_value),  # Best error achieved so far
                    }
                )

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

    # Log to WandB if configured
    if wandb_config:
        _log_to_wandb(wandb_config, problem, config, df, all_iteration_data)

    return df


def _log_to_wandb(
    wandb_config: Dict[str, Any],
    problem: OptimizationProblem,
    config: ExperimentConfig,
    df: pd.DataFrame,
    iteration_data: List[Dict],
):
    """Log experiment results to WandB using direct API."""
    if not WANDB_AVAILABLE:
        print("Warning: wandb not available, skipping logging")
        return

    # Create descriptive run name
    run_name = (
        f"protein_opt.analysis.{problem.name.lower()}_"
        f"{config.acquisition_fn}_"
        f"{'rand' if config.randomize_acquisition else 'det'}_"
        f"{len(config.seeds)}seeds_"
        f"{config.total_iterations}iter"
    )

    # Initialize wandb run directly
    run = wandb.init(
        project=wandb_config["project"],
        entity=wandb_config.get("entity"),
        name=run_name,
        tags=["sweep", "protein-opt-diagnostics"],
        group="protein-opt-diagnostics",
        config={
            "problem": problem.name,
            "problem_dim": problem.dim,
            "problem_optimum": problem.optimum_value,
            "acquisition_fn": config.acquisition_fn,
            "randomize_acquisition": config.randomize_acquisition,
            "num_seeds": len(config.seeds),
            "total_iterations": config.total_iterations,
            "stages": [
                {
                    "name": s.name,
                    "iterations": s.iterations,
                    "ucb_beta": s.ucb_beta,
                    "expansion_rate": s.expansion_rate,
                }
                for s in config.stages
            ],
        },
    )

    try:
        # Log iteration-level metrics
        _log_wandb_metrics(run, iteration_data, config, problem)
        
        # Create and log confidence interval plot
        _log_error_ci_plot(run, iteration_data, config, problem)

        # Log final summary metrics
        run.summary.update(
            {
                "final_best_median": df["best_median"].iloc[-1],
                "final_error": abs(df["best_median"].iloc[-1] - problem.optimum_value),
                "total_time_median": df["elapsed_median"].sum() if "elapsed_median" in df else 0,
                "convergence_iteration": _find_convergence_iteration(df, problem.optimum_value),
            }
        )
    finally:
        # Ensure run is properly closed
        run.finish()


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
            # Error statistics (current iteration)
            "error/median": iter_data["error"].median(),
            "error/mean": iter_data["error"].mean(),
            "error/std": iter_data["error"].std(),
            "error/min": iter_data["error"].min(),
            "error/max": iter_data["error"].max(),
            # Best error statistics (best so far)
            "best_error/median": iter_data["best_error"].median(),
            "best_error/mean": iter_data["best_error"].mean(),
            "best_error/std": iter_data["best_error"].std(),
            "best_error/min": iter_data["best_error"].min(),
            "best_error/max": iter_data["best_error"].max(),
            # Time statistics
            "time/median": iter_data["elapsed"].median(),
            "time/mean": iter_data["elapsed"].mean(),
            # Legacy metrics (kept for compatibility)
            "error/median_old": abs(iter_data["best_so_far"].median() - problem.optimum_value),
            "error/min_old": abs(iter_data["best_so_far"].min() - problem.optimum_value),
        }

        # Add per-seed values for detailed tracking
        for seed in config.seeds:
            seed_data = iter_data[iter_data["seed"] == seed]
            if len(seed_data) > 0:
                metrics[f"seed_{seed}/value"] = seed_data["value"].iloc[0]
                metrics[f"seed_{seed}/best"] = seed_data["best_so_far"].iloc[0]
                metrics[f"seed_{seed}/error"] = seed_data["error"].iloc[0]
                metrics[f"seed_{seed}/best_error"] = seed_data["best_error"].iloc[0]

        wandb_run.log(metrics, step=iteration)


def _log_error_ci_plot(wandb_run, iteration_data: List[Dict], config: ExperimentConfig, problem: OptimizationProblem):
    """Create and log a confidence interval plot for error over iterations."""
    if not iteration_data:
        return
    
    # Convert to DataFrame for easier manipulation
    iter_df = pd.DataFrame(iteration_data)
    
    # Prepare data for plotting
    iterations = []
    medians = []
    p25s = []
    p75s = []
    p10s = []
    p90s = []
    
    for iteration in range(1, config.total_iterations + 1):
        iter_data = iter_df[iter_df["iteration"] == iteration]
        if len(iter_data) > 0:
            iterations.append(iteration)
            best_errors = iter_data["best_error"].values
            
            # Calculate percentiles
            medians.append(np.median(best_errors))
            p25s.append(np.percentile(best_errors, 25))
            p75s.append(np.percentile(best_errors, 75))
            p10s.append(np.percentile(best_errors, 10))
            p90s.append(np.percentile(best_errors, 90))
    
    # Create the plot data
    plot_data = [[x, y, p25, p75, p10, p90] for x, y, p25, p75, p10, p90 
                  in zip(iterations, medians, p25s, p75s, p10s, p90s)]
    
    # Create wandb table
    table = wandb.Table(data=plot_data, 
                       columns=["iteration", "median", "p25", "p75", "p10", "p90"])
    
    # Log the custom plot with confidence intervals
    wandb_run.log({
        "error_confidence_plot": wandb.plot.line_series(
            xs=iterations,
            ys=[medians, p25s, p75s, p10s, p90s],
            keys=["Median", "25th percentile", "75th percentile", "10th percentile", "90th percentile"],
            title=f"Error Convergence with Confidence Intervals - {problem.name}",
            xname="Iteration"
        )
    })
    
    # Also create a custom chart with shaded regions
    try:
        import plotly.graph_objects as go
        PLOTLY_AVAILABLE = True
    except ImportError:
        PLOTLY_AVAILABLE = False
        print("Warning: plotly not available, skipping shaded CI plot")
        return
    
    fig = go.Figure()
    
    # Add 10-90% confidence interval (lightest shade)
    fig.add_trace(go.Scatter(
        x=iterations + iterations[::-1],
        y=p10s + p90s[::-1],
        fill='toself',
        fillcolor='rgba(0,100,200,0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        name='10-90% CI',
        showlegend=True
    ))
    
    # Add 25-75% confidence interval (medium shade)
    fig.add_trace(go.Scatter(
        x=iterations + iterations[::-1],
        y=p25s + p75s[::-1],
        fill='toself',
        fillcolor='rgba(0,100,200,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='25-75% CI',
        showlegend=True
    ))
    
    # Add median line
    fig.add_trace(go.Scatter(
        x=iterations,
        y=medians,
        line=dict(color='rgb(0,100,200)', width=2),
        name='Median',
        showlegend=True
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Error Convergence: {problem.name} - {config.acquisition_fn}',
        xaxis_title='Iteration',
        yaxis_title='Error from Optimum',
        yaxis_type='log',  # Log scale for error
        hovermode='x unified',
        template='plotly_white'
    )
    
    # Log the plotly figure
    wandb_run.log({"error_ci_plot": wandb.Plotly(fig)})


def _find_convergence_iteration(df: pd.DataFrame, optimum_value: float, tolerance: float = 0.01) -> int:
    """Find the iteration where the median best value converges to within tolerance of optimum."""
    if "best_median" not in df:
        return -1

    for i, val in enumerate(df["best_median"]):
        if abs(val - optimum_value) <= tolerance:
            return i + 1

    return -1


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
  standard:      Single balanced stage with 70 iterations
  adaptive:      Three stages - explore(30) -> balance(20) -> exploit(20)
  fast-adaptive: Three stages - explore(18) -> balance(11) -> exploit(11) [40 total]
  custom:        Single stage with 50 iterations (faster for testing)

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
        choices=["standard", "adaptive", "custom", "fast-adaptive"],
        help="Optimization stage configuration - see above for details (default: standard)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress during optimization (default: False)",
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
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Number of parallel workers for seed execution (default: auto-detect CPUs)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Disable parallel execution (useful for debugging)",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate seeds
    seeds = list(range(42, 42 + args.nseeds))

    # Determine number of workers
    if args.sequential:
        n_workers = 1
    else:
        n_workers = args.n_workers
        # If n_workers specified, ensure num_seeds is divisible by it
        if n_workers is not None and len(seeds) % n_workers != 0:
            # Adjust seeds to be divisible by n_workers
            adjusted_seeds = len(seeds) - (len(seeds) % n_workers)
            if adjusted_seeds < n_workers:
                adjusted_seeds = n_workers
            print(
                f"Warning: Adjusting number of seeds from {len(seeds)} to {adjusted_seeds} to be divisible by {n_workers} workers"
            )
            seeds = list(range(42, 42 + adjusted_seeds))

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
    elif args.stages == "fast-adaptive":
        # Fast three-stage optimization with same ratios as adaptive but 40 total iterations
        # Ratios: 30/70 = 0.43, 20/70 = 0.29, 20/70 = 0.29
        # Applied to 40: 0.43*40 = 17, 0.29*40 = 11.5, 0.29*40 = 11.5
        # Rounding to 18, 11, 11 to sum to 40
        stages = [
            OptimizationStage(
                name="explore", iterations=18, ucb_beta=3.0, expansion_rate=0.5, suggestions_per_pareto=128
            ),
            OptimizationStage(
                name="balance", iterations=11, ucb_beta=2.0, expansion_rate=0.25, suggestions_per_pareto=64
            ),
            OptimizationStage(
                name="exploit", iterations=11, ucb_beta=1.0, expansion_rate=0.1, suggestions_per_pareto=32
            ),
        ]
    else:  # custom
        # You can define custom stages here
        stages = [OptimizationStage(name="full", iterations=50)]

    # Set up wandb config if enabled
    wandb_config = None
    if args.wandb:
        if not WANDB_AVAILABLE:
            print("Error: wandb requested but not installed. Install with: pip install wandb")
            return 1
        wandb_config = {
            "project": args.wandb_project,
            "entity": args.wandb_entity,
        }

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

                df = run_experiment(
                    problem, config, problem_dir, verbose=args.verbose, wandb_config=wandb_config, n_workers=n_workers
                )
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
