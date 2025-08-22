"""Hartmann6 optimization using PROTEIN with tAXIOM - Multiseed version.

This example demonstrates using tAXIOM to orchestrate PROTEIN optimization
with multiple seeds for statistical robustness.
"""

import time
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel

from metta.sweep.axiom import Ctx, Pipeline, context_aware
from metta.sweep.protein import Protein

# ============================================================================
# Domain Models
# ============================================================================


class Phase(BaseModel):
    """Optimization phase configuration."""

    name: str
    expansion_rate: float = 0.25
    suggestions_per_pareto: int = 64


class Trial(BaseModel):
    """Single optimization trial."""

    id: int
    seed: int
    params: Dict[str, float]
    score: Optional[float] = None
    phase: Optional[str] = None
    elapsed: float = 0.0


class SeedRun(BaseModel):
    """Results from a single seed."""

    seed: int
    trials: List[Trial]
    best_score: float
    best_params: Dict[str, float]
    total_time: float
    convergence: List[float]  # Best score at each iteration


class MultiSeedSummary(BaseModel):
    """Aggregated results from multiple seeds."""

    seeds: List[int]
    runs: List[SeedRun]

    def aggregate_stats(self) -> Dict:
        """Calculate aggregate statistics across seeds."""
        all_best_scores = [run.best_score for run in self.runs]

        # Convergence curves for each seed
        convergence_curves = [run.convergence for run in self.runs]
        max_len = max(len(c) for c in convergence_curves)

        # Pad curves to same length
        padded_curves = []
        for curve in convergence_curves:
            padded = curve + [curve[-1]] * (max_len - len(curve))
            padded_curves.append(padded)

        # Calculate percentiles at each iteration
        curves_array = np.array(padded_curves)
        median_curve = np.median(curves_array, axis=0)
        p25_curve = np.percentile(curves_array, 25, axis=0)
        p75_curve = np.percentile(curves_array, 75, axis=0)

        return {
            "best_scores": all_best_scores,
            "median_best": np.median(all_best_scores),
            "mean_best": np.mean(all_best_scores),
            "std_best": np.std(all_best_scores),
            "min_best": np.min(all_best_scores),
            "max_best": np.max(all_best_scores),
            "median_convergence": median_curve.tolist(),
            "p25_convergence": p25_curve.tolist(),
            "p75_convergence": p75_curve.tolist(),
        }


class MultiSeedAnalysis(BaseModel):
    """Analysis of multiseed experiment."""

    median_best: float
    mean_best: float
    std_best: float
    gap_to_optimum: float
    relative_error: float
    success_rate: float  # Fraction of seeds within threshold
    convergence_speed: int  # Median iterations to reach threshold


# ============================================================================
# Hartmann6 Test Function
# ============================================================================


def hartmann6(x: np.ndarray) -> float:
    """6D Hartmann function. Global minimum: -3.32237."""
    alpha = np.array([1.0, 1.2, 3.0, 3.2])

    A = np.array(
        [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ]
    )

    P = 1e-4 * np.array(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ]
    )

    outer_sum = 0
    for i in range(4):
        inner_sum = sum(A[i, j] * (x[j] - P[i, j]) ** 2 for j in range(6))
        outer_sum += alpha[i] * np.exp(-inner_sum)

    return -outer_sum


# ============================================================================
# Single Trial Pipeline Stages (same as before, but seed-aware)
# ============================================================================


@context_aware
def choose_phase(ctx: Ctx) -> Phase:
    """Select optimization phase based on trial count."""
    trial_id = ctx.metadata.get("trial_id", 0)

    # Simple 3-phase schedule with expansion rate for naive acquisition
    if trial_id < 30:
        return Phase(name="explore", expansion_rate=0.5, suggestions_per_pareto=128)
    elif trial_id < 50:
        return Phase(name="balance", expansion_rate=0.25, suggestions_per_pareto=64)
    else:
        return Phase(name="exploit", expansion_rate=0.1, suggestions_per_pareto=32)


@context_aware
def load_optimizer(ctx: Ctx) -> Protein:
    """Load or create PROTEIN optimizer."""
    # Get phase from previous stage output
    phase = ctx.get_stage_output("choose_phase")
    if phase is None:
        phase = Phase(name="balance", expansion_rate=0.25)

    # Try to get existing optimizer
    optimizer = ctx.metadata.get("optimizer")

    if optimizer is None:
        # Create new optimizer
        sweep_config = {
            "metric": "hartmann6",
            "goal": "minimize",
        }

        # Add parameter spaces
        for i in range(6):
            sweep_config[f"x{i}"] = {
                "distribution": "uniform",
                "min": 0.0,
                "max": 1.0,
                "mean": 0.5,
                "scale": "auto",
            }

        optimizer = Protein(
            sweep_config=sweep_config,
            acquisition_fn="naive",  # Using PROTEIN's original acquisition function
            expansion_rate=phase.expansion_rate,
            suggestions_per_pareto=phase.suggestions_per_pareto,
        )

        # Replay observations if any
        trials_data = ctx.metadata.get("trials", [])
        for trial_dict in trials_data:
            if trial_dict.get("score") is not None:
                optimizer.observe(trial_dict["params"], trial_dict["score"], cost=1.0)
    else:
        # Update existing optimizer with new phase params
        optimizer.expansion_rate = phase.expansion_rate
        optimizer.suggestions_per_pareto = phase.suggestions_per_pareto

    # Store optimizer back
    ctx.metadata["optimizer"] = optimizer
    return optimizer


def suggest(optimizer: Protein) -> Dict[str, float]:
    """Get next suggestion from optimizer."""
    params, _ = optimizer.suggest()
    return params


@context_aware
def evaluate_with_context(ctx: Ctx) -> Trial:
    """Evaluate parameters and create trial."""
    # Get params from previous stage
    params = ctx.get_stage_output("suggest")
    if params is None:
        params = {f"x{i}": 0.5 for i in range(6)}

    start = time.time()

    # Evaluate function
    x = np.array([params[f"x{i}"] for i in range(6)])
    score = hartmann6(x)

    # Get current phase and seed
    phase = ctx.get_stage_output("choose_phase")
    seed = ctx.metadata.get("current_seed", 0)

    # Create trial
    trial = Trial(
        id=ctx.metadata.get("trial_id", 0),
        seed=seed,
        params=params,
        score=score,
        phase=phase.name if phase else "unknown",
        elapsed=time.time() - start,
    )

    # Update optimizer
    optimizer = ctx.metadata.get("optimizer")
    if optimizer:
        optimizer.observe(params, score, cost=1.0)

    # Store trial as dict for serialization
    trials = ctx.metadata.get("trials", [])
    trials.append(trial.dict())
    ctx.metadata["trials"] = trials

    # Update best
    best_score = ctx.metadata.get("best_score", float("inf"))
    if score < best_score:
        ctx.metadata["best_score"] = score
        ctx.metadata["best_params"] = params

    # Track convergence
    convergence = ctx.metadata.get("convergence", [])
    convergence.append(ctx.metadata["best_score"])
    ctx.metadata["convergence"] = convergence

    # Increment trial counter
    ctx.metadata["trial_id"] = trial.id + 1

    return trial


def collect_seed_run(ctx: Ctx) -> SeedRun:
    """Collect results from a single seed run."""
    trials_data = ctx.metadata.get("trials", [])
    # Convert dicts back to Trial objects
    trials = [Trial(**t) for t in trials_data]

    seed = ctx.metadata.get("current_seed", 0)
    best_score = ctx.metadata.get("best_score", float("inf"))
    best_params = ctx.metadata.get("best_params", {})
    start_time = ctx.metadata.get("seed_start_time", time.time())
    convergence = ctx.metadata.get("convergence", [])

    run = SeedRun(
        seed=seed,
        trials=trials,
        best_score=best_score,
        best_params=best_params,
        total_time=time.time() - start_time,
        convergence=convergence,
    )

    print(f"Seed {seed} complete: best = {best_score:.5f} (time: {run.total_time:.1f}s)")

    return run


# ============================================================================
# Aggregation and Analysis Stages
# ============================================================================


def analyze_multiseed(summary: MultiSeedSummary) -> MultiSeedAnalysis:
    """Analyze multiseed results."""
    optimum = -3.32237
    threshold = 0.1  # Success threshold

    stats = summary.aggregate_stats()

    # Calculate success rate
    successes = sum(1 for run in summary.runs if abs(run.best_score - optimum) < threshold)
    success_rate = successes / len(summary.runs) if summary.runs else 0

    # Calculate convergence speed (median iterations to reach within threshold)
    convergence_iters = []
    for run in summary.runs:
        for i, score in enumerate(run.convergence):
            if abs(score - optimum) < threshold:
                convergence_iters.append(i + 1)
                break
        else:
            convergence_iters.append(len(run.convergence))

    convergence_speed = int(np.median(convergence_iters)) if convergence_iters else 0

    return MultiSeedAnalysis(
        median_best=stats["median_best"],
        mean_best=stats["mean_best"],
        std_best=stats["std_best"],
        gap_to_optimum=abs(stats["median_best"] - optimum),
        relative_error=abs(stats["median_best"] - optimum) / abs(optimum),
        success_rate=success_rate,
        convergence_speed=convergence_speed,
    )


def report_multiseed(analysis: MultiSeedAnalysis) -> None:
    """Print multiseed experiment report."""
    print("\n" + "=" * 60)
    print("Hartmann6 Multi-Seed Results")
    print("=" * 60)
    print(f"Median best:    {analysis.median_best:.5f} ± {analysis.std_best:.5f}")
    print(f"Mean best:      {analysis.mean_best:.5f}")
    print("Global optimum: -3.32237")
    print(f"Gap:            {analysis.gap_to_optimum:.5f}")
    print(f"Relative error: {analysis.relative_error:.2%}")
    print(f"Success rate:   {analysis.success_rate:.1%} (within 0.1 of optimum)")
    print(f"Convergence:    {analysis.convergence_speed} iterations (median)")

    # Success message
    if analysis.gap_to_optimum < 0.01:
        print("\n✓ Excellent! Median within 0.01 of optimum")
    elif analysis.gap_to_optimum < 0.1:
        print("\n✓ Good! Median within 0.1 of optimum")
    else:
        print(f"\n⚠ Gap of {analysis.gap_to_optimum:.3f} from optimum")


# ============================================================================
# Pipeline Construction
# ============================================================================

"""
Data Flow Through Multi-Seed Pipeline:

1. Seed Loop:
   ∅ → schedule_seed → int
   int → setup_seed_run → Dict
   
2. Trial Loop (per seed):
   ∅ → choose_phase → Phase
   Phase → load_optimizer → Protein
   Protein → suggest → Dict[str, float]
   Dict[str, float] → evaluate → Trial
   
3. Aggregation:
   [SeedRun] → aggregate_runs → MultiSeedSummary
   MultiSeedSummary → analyze → MultiSeedAnalysis
   MultiSeedAnalysis → report → None
"""


def build_multiseed_pipeline(seeds: List[int], n_trials_per_seed: int = 70) -> Pipeline:
    """Build the multi-seed optimization pipeline."""

    # Single trial pipeline
    trial_pipeline = (
        Pipeline()
        .stage("choose_phase", choose_phase)
        .through(Phase)
        .stage("load_optimizer", load_optimizer)
        .through(Protein)
        .stage("suggest", suggest)
        .through(Dict[str, float])
        .stage("evaluate", evaluate_with_context)
        .through(Trial)
    )

    # Main pipeline with seed loop
    main_pipeline = (
        Pipeline()
        .stage("run_all_seeds", lambda: run_all_seeds(trial_pipeline, seeds=seeds, n_trials=n_trials_per_seed))
        .through(MultiSeedSummary)
        .stage("analyze", analyze_multiseed)
        .through(MultiSeedAnalysis)
        .stage("report", report_multiseed)
        .through(type(None))
    )

    return main_pipeline


def run_single_seed(args: Tuple[Pipeline, int, int, int]) -> SeedRun:
    """Worker function to run optimization for a single seed."""
    pipeline, seed, seed_idx, n_trials = args

    # Create fresh context for this seed
    ctx = Ctx()
    ctx.metadata["current_seed"] = seed
    ctx.metadata["trial_id"] = 0
    ctx.metadata["optimizer"] = None
    ctx.metadata["trials"] = []
    ctx.metadata["seed_start_time"] = time.time()
    ctx.metadata["best_score"] = float("inf")
    ctx.metadata["best_params"] = {}
    ctx.metadata["convergence"] = []

    # Set random seed
    np.random.seed(seed)

    print(f"Worker starting seed {seed}...")

    # Run trials for this seed
    for i in range(n_trials):
        ctx.metadata["trial_id"] = i
        pipeline.run(ctx)

        # Progress update
        if (i + 1) % 20 == 0:
            best = ctx.metadata.get("best_score", float("inf"))
            print(f"  Seed {seed} - Trial {i + 1:3d}/{n_trials}: best = {best:.5f}")

    # Collect and return results
    return collect_seed_run(ctx)


def run_all_seeds(pipeline: Pipeline, seeds: List[int], n_trials: int) -> MultiSeedSummary:
    """Run optimization for all seeds in parallel."""
    # Determine number of workers
    n_workers = min(cpu_count(), len(seeds))

    # Adjust seeds to be divisible by workers
    if len(seeds) % n_workers != 0:
        # Find the best number of workers that divides evenly
        for w in range(n_workers, 0, -1):
            if len(seeds) % w == 0:
                n_workers = w
                break
        else:
            # If no perfect divisor, trim seeds
            n_seeds_adjusted = (len(seeds) // n_workers) * n_workers
            if n_seeds_adjusted > 0:
                print(f"Adjusting seeds from {len(seeds)} to {n_seeds_adjusted} for {n_workers} workers")
                seeds = seeds[:n_seeds_adjusted]

    print(f"\nRunning {len(seeds)} seeds with {n_workers} parallel workers")
    print(f"Seeds per worker: {len(seeds) // n_workers}")

    # Prepare arguments for parallel execution
    worker_args = [(pipeline, seed, idx, n_trials) for idx, seed in enumerate(seeds)]

    # Run in parallel
    with Pool(n_workers) as pool:
        runs = pool.map(run_single_seed, worker_args)

    # Create summary
    return MultiSeedSummary(seeds=seeds, runs=runs)


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """Run Hartmann6 optimization with multiple seeds."""
    print("Hartmann6 Multi-Seed Parallel Optimization with PROTEIN + tAXIOM")
    print("=" * 60)

    # Auto-detect CPUs and configure seeds
    n_cpus = cpu_count()
    print(f"Detected {n_cpus} CPU cores")

    # Define base number of seeds and adjust for parallel execution
    base_seeds = 20  # Desired number of seeds
    n_seeds = (base_seeds // n_cpus) * n_cpus  # Ensure divisible by CPU count

    if n_seeds == 0:
        n_seeds = n_cpus  # At least one per CPU

    # Generate seeds
    seeds = list(range(42, 42 + n_seeds))
    n_trials = 70  # Trials per seed

    print(f"Seeds: {n_seeds} seeds (adjusted from {base_seeds} for {n_cpus} CPUs)")
    print(f"Trials per seed: {n_trials}")
    print(f"Total evaluations: {n_seeds * n_trials}")
    print(f"Parallel workers: {min(n_cpus, n_seeds)}")

    # Build and run pipeline
    start_time = time.time()
    pipeline = build_multiseed_pipeline(seeds=seeds, n_trials_per_seed=n_trials)
    pipeline.run()

    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.1f}s")
    print(f"Time per seed (avg): {total_time / n_seeds:.1f}s")
    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
