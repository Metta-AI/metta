"""Parallel difficulty analysis for large-scale variant combination testing.

Runs many rollouts in parallel using multiprocessing for efficient CPU utilization.
Supports multiple repeats per combination for statistical power.

Usage:
    python -m cogames.cogs_vs_clips.parallel_difficulty_analysis \
        --combinations 1000 --repeats 10 --workers 16 --steps 1000
"""

from __future__ import annotations

import json
import multiprocessing as mp
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class RunConfig:
    """Configuration for a single run."""
    combo_id: int
    repeat_id: int
    variant_types: list[str]  # Class names to instantiate
    max_steps: int
    num_cogs: int
    seed: int


@dataclass
class RunResult:
    """Result from a single run."""
    combo_id: int
    repeat_id: int
    variant_names: list[str]
    estimated_hearts_1k: float
    estimated_difficulty: float
    actual_hearts: float
    actual_steps: int
    success: bool
    error: str | None = None


def _run_single(config: RunConfig) -> RunResult:
    """Run a single evaluation (called in worker process)."""
    try:
        # Import inside worker to avoid pickling issues
        from cogames.cogs_vs_clips.difficulty_estimator import estimate_difficulty
        from cogames.cogs_vs_clips.mission import Mission, NumCogsVariant
        from cogames.cogs_vs_clips.sites import HELLO_WORLD
        from cogames.cogs_vs_clips import variants as v

        # Instantiate variants from class names
        variant_instances = []
        for vtype_name in config.variant_types:
            vtype = getattr(v, vtype_name, None)
            if vtype is not None:
                variant_instances.append(vtype())

        # Create mission
        mission = Mission(
            name=f"combo_{config.combo_id}",
            description="Parallel test",
            site=HELLO_WORLD,
            variants=variant_instances,
        )
        mission_with_cogs = mission.with_variants([NumCogsVariant(num_cogs=config.num_cogs)])

        # Get estimate
        report = estimate_difficulty(mission_with_cogs)
        estimated_hearts_1k = (
            config.max_steps / report.expected_steps_per_heart
            if report.expected_steps_per_heart > 0
            else 0
        )

        # Run thinky agents
        from cogames.cogs_vs_clips.difficulty_evaluation import run_thinky_evaluation
        actual_hearts, actual_steps = run_thinky_evaluation(
            mission_with_cogs, config.max_steps, config.num_cogs, config.seed
        )

        return RunResult(
            combo_id=config.combo_id,
            repeat_id=config.repeat_id,
            variant_names=[vi.name for vi in variant_instances],
            estimated_hearts_1k=estimated_hearts_1k,
            estimated_difficulty=report.difficulty_score,
            actual_hearts=actual_hearts,
            actual_steps=actual_steps,
            success=True,
        )

    except Exception as e:
        return RunResult(
            combo_id=config.combo_id,
            repeat_id=config.repeat_id,
            variant_names=config.variant_types,
            estimated_hearts_1k=0,
            estimated_difficulty=float("inf"),
            actual_hearts=0,
            actual_steps=0,
            success=False,
            error=str(e),
        )


def _init_worker():
    """Initialize worker process - suppress noisy logs."""
    import logging
    logging.getLogger().setLevel(logging.WARNING)
    try:
        from cogames.cli.utils import suppress_noisy_logs
        suppress_noisy_logs()
    except ImportError:
        pass


@dataclass
class ParallelAnalysisResult:
    """Aggregated results from parallel analysis."""
    results: list[RunResult] = field(default_factory=list)
    total_time: float = 0.0
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0

    def to_json(self, path: str | Path) -> None:
        """Save results to JSON."""
        data = {
            "metadata": {
                "total_time": self.total_time,
                "total_runs": self.total_runs,
                "successful_runs": self.successful_runs,
                "failed_runs": self.failed_runs,
            },
            "results": [
                {
                    "combo_id": r.combo_id,
                    "repeat_id": r.repeat_id,
                    "variants": r.variant_names,
                    "est_hearts_1k": r.estimated_hearts_1k,
                    "est_difficulty": r.estimated_difficulty,
                    "actual_hearts": r.actual_hearts,
                    "actual_steps": r.actual_steps,
                    "success": r.success,
                    "error": r.error,
                }
                for r in self.results
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def summary(self) -> str:
        """Generate summary statistics."""
        lines = [
            "=" * 70,
            "PARALLEL DIFFICULTY ANALYSIS RESULTS",
            "=" * 70,
            f"Total runs: {self.total_runs} ({self.successful_runs} success, {self.failed_runs} failed)",
            f"Total time: {self.total_time:.1f}s ({self.total_runs / max(1, self.total_time):.1f} runs/sec)",
            "",
        ]

        # Filter successful with positive hearts
        valid = [r for r in self.results if r.success and r.actual_hearts > 0]
        if not valid:
            lines.append("No valid results with hearts > 0")
            return "\n".join(lines)

        # Compute statistics
        estimated = [r.estimated_hearts_1k for r in valid]
        actual = [r.actual_hearts for r in valid]
        ratios = [a / e for a, e in zip(actual, estimated) if e > 0]

        corr = float(np.corrcoef(estimated, actual)[0, 1]) if len(valid) > 2 else 0.0

        lines.extend([
            f"Valid samples (hearts > 0): {len(valid)}",
            f"Correlation (est vs actual): r = {corr:.3f}",
            f"Median ratio (actual/est): {np.median(ratios):.3f}",
            f"Mean ratio: {np.mean(ratios):.3f}",
            f"Std ratio: {np.std(ratios):.3f}",
            "",
        ])

        # Per-combination statistics (average across repeats)
        from collections import defaultdict
        combo_stats: dict[int, list[float]] = defaultdict(list)
        combo_variants: dict[int, list[str]] = {}

        for r in valid:
            combo_stats[r.combo_id].append(r.actual_hearts)
            combo_variants[r.combo_id] = r.variant_names

        lines.append("Top 15 combinations by mean actual hearts:")
        sorted_combos = sorted(combo_stats.items(), key=lambda x: -np.mean(x[1]))[:15]
        for combo_id, hearts_list in sorted_combos:
            mean_h = np.mean(hearts_list)
            std_h = np.std(hearts_list)
            variants = combo_variants[combo_id]
            vname = "+".join(variants[:3]) or "base"
            if len(variants) > 3:
                vname += "..."
            lines.append(f"  {mean_h:>5.1f}♥ ± {std_h:>4.1f} | {vname}")

        # Zero hearts analysis
        zeros = [r for r in self.results if r.success and r.actual_hearts == 0]
        if zeros:
            lines.extend(["", f"Zero hearts runs: {len(zeros)}"])
            zero_variants: dict[str, int] = defaultdict(int)
            for r in zeros:
                for v in r.variant_names:
                    zero_variants[v] += 1
            top_zeros = sorted(zero_variants.items(), key=lambda x: -x[1])[:5]
            for v, count in top_zeros:
                lines.append(f"  {v}: {count} ({count/len(zeros)*100:.0f}%)")

        return "\n".join(lines)


def generate_combinations(
    n_combinations: int,
    seed: int = 42,
) -> list[list[str]]:
    """Generate unique variant combinations as class name lists."""
    from cogames.cogs_vs_clips.variant_shuffler import COMBINABLE_VARIANTS, check_conflicts

    rng = random.Random(seed)
    combinations: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()

    # Always include base (no variants)
    combinations.append([])
    seen.add(())

    attempts = 0
    max_attempts = n_combinations * 50

    while len(combinations) < n_combinations and attempts < max_attempts:
        attempts += 1

        num_v = rng.randint(0, min(5, len(COMBINABLE_VARIANTS)))
        if num_v == 0:
            variant_types: list[type] = []
        else:
            variant_types = rng.sample(COMBINABLE_VARIANTS, num_v)

        if check_conflicts(variant_types):
            continue

        key = tuple(sorted(v.__name__ for v in variant_types))
        if key in seen:
            continue

        seen.add(key)
        combinations.append([v.__name__ for v in variant_types])

    return combinations


def run_parallel_analysis(
    n_combinations: int = 100,
    n_repeats: int = 5,
    max_steps: int = 1000,
    num_cogs: int = 4,
    n_workers: int | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> ParallelAnalysisResult:
    """Run parallel difficulty analysis.

    Args:
        n_combinations: Number of unique variant combinations to test
        n_repeats: Number of repeat runs per combination
        max_steps: Max steps per episode
        num_cogs: Number of agents
        n_workers: Number of parallel workers (default: CPU count)
        seed: Random seed
        verbose: Print progress

    Returns:
        ParallelAnalysisResult with all results
    """
    if n_workers is None:
        n_workers = mp.cpu_count()

    if verbose:
        print(f"Generating {n_combinations} unique combinations...")

    combinations = generate_combinations(n_combinations, seed)
    actual_combos = len(combinations)

    if verbose:
        print(f"Generated {actual_combos} unique combinations")
        print(f"Running {actual_combos} × {n_repeats} = {actual_combos * n_repeats} total runs")
        print(f"Using {n_workers} parallel workers")

    # Create run configs
    configs: list[RunConfig] = []
    for combo_id, variant_types in enumerate(combinations):
        for repeat_id in range(n_repeats):
            configs.append(RunConfig(
                combo_id=combo_id,
                repeat_id=repeat_id,
                variant_types=variant_types,
                max_steps=max_steps,
                num_cogs=num_cogs,
                seed=seed + combo_id * 1000 + repeat_id,
            ))

    # Run in parallel
    start_time = time.time()
    results: list[RunResult] = []

    with mp.Pool(n_workers, initializer=_init_worker) as pool:
        if verbose:
            # Use imap for progress tracking
            for i, result in enumerate(pool.imap_unordered(_run_single, configs)):
                results.append(result)
                if (i + 1) % 50 == 0 or i + 1 == len(configs):
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    eta = (len(configs) - i - 1) / rate if rate > 0 else 0
                    print(f"[{i+1:>5}/{len(configs)}] {rate:.1f} runs/sec, ETA {eta:.0f}s")
        else:
            results = pool.map(_run_single, configs)

    total_time = time.time() - start_time

    # Build result
    analysis = ParallelAnalysisResult(
        results=results,
        total_time=total_time,
        total_runs=len(results),
        successful_runs=sum(1 for r in results if r.success),
        failed_runs=sum(1 for r in results if not r.success),
    )

    return analysis


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Parallel Difficulty Analysis")
    parser.add_argument("-n", "--combinations", type=int, default=100, help="Number of combinations")
    parser.add_argument("-r", "--repeats", type=int, default=5, help="Repeats per combination")
    parser.add_argument("--steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--cogs", type=int, default=4, help="Number of agents")
    parser.add_argument("-w", "--workers", type=int, default=None, help="Number of workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("-o", "--output", default="parallel_results.json", help="Output file")
    parser.add_argument("-q", "--quiet", action="store_true", help="Quiet mode")

    args = parser.parse_args()

    result = run_parallel_analysis(
        n_combinations=args.combinations,
        n_repeats=args.repeats,
        max_steps=args.steps,
        num_cogs=args.cogs,
        n_workers=args.workers,
        seed=args.seed,
        verbose=not args.quiet,
    )

    result.to_json(args.output)
    print(f"\nSaved to: {args.output}")
    print(result.summary())


if __name__ == "__main__":
    main()


