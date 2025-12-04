"""Unified difficulty analysis system for Cogs vs Clips missions.

Provides tools to:
1. Estimate mission difficulty (fast oracle-based estimator)
2. Evaluate actual agent performance (thinky agents)
3. Compare estimates vs actual
4. Analyze variant interactions
5. Generate visualizations

Usage:
    # Estimate a single mission
    from cogames.cogs_vs_clips.difficulty_analysis import analyze_mission
    report = analyze_mission(HelloWorldOpenWorldMission)

    # Run comparison across variants
    from cogames.cogs_vs_clips.difficulty_analysis import run_comparison
    results = run_comparison(n_samples=50)

    # CLI usage
    python -m cogames.cogs_vs_clips.difficulty_analysis estimate HelloWorldOpenWorldMission
    python -m cogames.cogs_vs_clips.difficulty_analysis compare --samples 50
    python -m cogames.cogs_vs_clips.difficulty_analysis interactions
"""

from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class MissionAnalysis:
    """Complete analysis of a single mission."""

    name: str
    variants: list[str]

    # Fast estimator results
    estimated_difficulty: float = 0.0
    estimated_steps_per_heart: float = 0.0
    estimated_hearts_1k: float = 0.0
    exploration_steps: int = 0
    feasible: bool = True

    # Thinky agent results (if evaluated)
    actual_hearts: float | None = None
    actual_steps: int | None = None

    @property
    def estimation_ratio(self) -> float | None:
        """Ratio of estimated to actual (< 1 = underestimate)."""
        if self.actual_hearts and self.actual_hearts > 0:
            return self.estimated_hearts_1k / self.actual_hearts
        return None

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Mission: {self.name}",
            f"Variants: {', '.join(self.variants) or 'base'}",
            "",
            f"Estimated Difficulty: {self.estimated_difficulty:.3f}",
            f"Estimated Steps/Heart: {self.estimated_steps_per_heart:.0f}",
            f"Estimated Hearts (1k steps): {self.estimated_hearts_1k:.1f}",
            f"Exploration: {self.exploration_steps} steps",
            f"Feasible: {'Yes' if self.feasible else 'No'}",
        ]

        if self.actual_hearts is not None:
            lines.extend([
                "",
                f"Actual Hearts: {self.actual_hearts:.0f}",
                f"Actual Steps: {self.actual_steps}",
                f"Est/Actual Ratio: {self.estimation_ratio:.2f}x" if self.estimation_ratio else "N/A",
            ])

        return "\n".join(lines)


@dataclass
class ComparisonResults:
    """Results from comparing estimates vs actual across multiple missions."""

    analyses: list[MissionAnalysis] = field(default_factory=list)

    @property
    def correlation(self) -> float:
        """Correlation between estimated and actual hearts."""
        valid = [a for a in self.analyses if a.actual_hearts and a.actual_hearts > 0]
        if len(valid) < 3:
            return 0.0
        estimated = [a.estimated_hearts_1k for a in valid]
        actual = [a.actual_hearts for a in valid]
        return float(np.corrcoef(estimated, actual)[0, 1])

    @property
    def median_ratio(self) -> float:
        """Median of actual/estimated ratio."""
        ratios = [a.actual_hearts / a.estimated_hearts_1k
                  for a in self.analyses
                  if a.actual_hearts and a.actual_hearts > 0 and a.estimated_hearts_1k > 0]
        return float(np.median(ratios)) if ratios else 0.0

    def summary(self) -> str:
        """Summary statistics."""
        lines = [
            "=" * 60,
            "COMPARISON SUMMARY",
            "=" * 60,
            f"Total missions: {len(self.analyses)}",
            f"Correlation (est vs actual): r = {self.correlation:.3f}",
            f"Median ratio (actual/est): {self.median_ratio:.2f}",
            "",
            "Top performers by actual hearts:",
        ]

        sorted_by_actual = sorted(
            [a for a in self.analyses if a.actual_hearts],
            key=lambda x: -(x.actual_hearts or 0)
        )[:10]

        for a in sorted_by_actual:
            ratio = f"{a.estimation_ratio:.1f}x" if a.estimation_ratio else "N/A"
            lines.append(f"  {a.actual_hearts:>5.0f}♥ (est {a.estimated_hearts_1k:>5.0f}) {ratio:>6} | {'+'.join(a.variants[:2]) or 'base'}")

        return "\n".join(lines)

    def to_json(self, path: str | Path) -> None:
        """Save results to JSON."""
        data = [
            {
                "name": a.name,
                "variants": a.variants,
                "est_difficulty": a.estimated_difficulty,
                "est_steps_per_heart": a.estimated_steps_per_heart,
                "est_hearts_1k": a.estimated_hearts_1k,
                "actual_hearts": a.actual_hearts,
                "actual_steps": a.actual_steps,
            }
            for a in self.analyses
        ]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


@dataclass
class InteractionAnalysis:
    """Analysis of variant interactions."""

    variants: list[str]
    single_effects: dict[str, float]  # variant -> effect on hearts
    interaction_matrix: np.ndarray  # pairwise interactions

    @property
    def top_synergies(self) -> list[tuple[str, str, float]]:
        """Top synergistic variant pairs."""
        pairs = []
        n = len(self.variants)
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((self.variants[i], self.variants[j], self.interaction_matrix[i, j]))
        pairs.sort(key=lambda x: -x[2])
        return pairs[:5]

    @property
    def top_antagonisms(self) -> list[tuple[str, str, float]]:
        """Top antagonistic variant pairs."""
        pairs = []
        n = len(self.variants)
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((self.variants[i], self.variants[j], self.interaction_matrix[i, j]))
        pairs.sort(key=lambda x: x[2])
        return pairs[:5]


# =============================================================================
# Core Analysis Functions
# =============================================================================


def analyze_mission(
    mission,
    run_thinky: bool = False,
    max_steps: int = 1000,
    num_cogs: int = 4,
    seed: int = 42,
) -> MissionAnalysis:
    """Analyze a single mission.

    Args:
        mission: Mission instance to analyze
        run_thinky: Whether to run thinky agents for actual performance
        max_steps: Max steps for thinky evaluation
        num_cogs: Number of agents
        seed: Random seed

    Returns:
        MissionAnalysis with estimates and optionally actual performance
    """
    from cogames.cogs_vs_clips.difficulty_estimator import estimate_difficulty
    from cogames.cogs_vs_clips.mission import NumCogsVariant

    # Apply num_cogs
    mission_with_cogs = mission.with_variants([NumCogsVariant(num_cogs=num_cogs)])

    # Get fast estimate
    report = estimate_difficulty(mission_with_cogs)

    analysis = MissionAnalysis(
        name=mission.name,
        variants=[v.name for v in mission.variants if hasattr(v, "name")],
        estimated_difficulty=report.difficulty_score,
        estimated_steps_per_heart=report.expected_steps_per_heart,
        estimated_hearts_1k=max_steps / report.expected_steps_per_heart if report.expected_steps_per_heart > 0 else 0,
        exploration_steps=report.exploration.multi_agent_discovery if report.exploration else 0,
        feasible=report.feasible,
    )

    # Optionally run thinky agents
    if run_thinky:
        from cogames.cogs_vs_clips.difficulty_evaluation import run_thinky_evaluation
        actual_hearts, steps = run_thinky_evaluation(mission_with_cogs, max_steps, num_cogs, seed)
        analysis.actual_hearts = actual_hearts
        analysis.actual_steps = steps

    return analysis


def run_comparison(
    n_samples: int = 50,
    max_steps: int = 1000,
    num_cogs: int = 4,
    seed: int = 42,
    verbose: bool = True,
) -> ComparisonResults:
    """Run comparison across random variant combinations.

    Args:
        n_samples: Number of random combinations to test
        max_steps: Max steps per evaluation
        num_cogs: Number of agents
        seed: Random seed
        verbose: Print progress

    Returns:
        ComparisonResults with all analyses
    """
    from cogames.cogs_vs_clips.mission import Mission
    from cogames.cogs_vs_clips.sites import HELLO_WORLD
    from cogames.cogs_vs_clips.variant_shuffler import COMBINABLE_VARIANTS, check_conflicts

    # Base mission - no hardcoded variants, let the shuffler add them
    base_mission = Mission(
        name="comparison_base",
        description="Base for comparison",
        site=HELLO_WORLD,
        variants=[],
    )

    # Generate random combinations
    rng = random.Random(seed)
    combinations = []
    seen = set()

    attempts = 0
    while len(combinations) < n_samples and attempts < n_samples * 20:
        attempts += 1
        num_v = rng.randint(0, 4)
        variant_types = rng.sample(COMBINABLE_VARIANTS, min(num_v, len(COMBINABLE_VARIANTS))) if num_v > 0 else []

        if check_conflicts(variant_types):
            continue

        key = tuple(sorted(v.__name__ for v in variant_types))
        if key in seen:
            continue
        seen.add(key)
        combinations.append(variant_types)

    if verbose:
        print(f"Generated {len(combinations)} unique combinations")

    # Run analyses
    results = ComparisonResults()
    start_time = time.time()

    for i, variant_types in enumerate(combinations):
        variants = [v() for v in variant_types]
        mission = base_mission.with_variants(variants)

        if verbose:
            name = "+".join(v.name for v in variants)[:40] or "base"
            print(f"[{i+1:>3}/{len(combinations)}] {name:<40}", end=" ", flush=True)

        analysis = analyze_mission(mission, run_thinky=True, max_steps=max_steps, num_cogs=num_cogs, seed=seed + i)
        results.analyses.append(analysis)

        if verbose:
            print(f"Est={analysis.estimated_hearts_1k:>5.0f} Act={analysis.actual_hearts or 0:>5.0f}")

    if verbose:
        elapsed = time.time() - start_time
        print(f"\nCompleted in {elapsed:.1f}s")

    return results


def analyze_interactions(results: ComparisonResults) -> InteractionAnalysis:
    """Analyze variant interactions from comparison results.

    Args:
        results: ComparisonResults from run_comparison

    Returns:
        InteractionAnalysis with single effects and interaction matrix
    """
    # Get all variants
    all_variants = set()
    for a in results.analyses:
        all_variants.update(a.variants)
    variants = sorted(all_variants)

    if not variants:
        return InteractionAnalysis(variants=[], single_effects={}, interaction_matrix=np.array([]))

    # Calculate baseline
    baseline_analyses = [a for a in results.analyses if not a.variants]
    baseline = np.mean([a.actual_hearts or 0 for a in baseline_analyses]) if baseline_analyses else 5.0

    # Calculate single effects
    single_effects = {}
    for v in variants:
        with_v = [a.actual_hearts or 0 for a in results.analyses if v in a.variants]
        without_v = [a.actual_hearts or 0 for a in results.analyses if v not in a.variants]
        if with_v and without_v:
            single_effects[v] = np.mean(with_v) - np.mean(without_v)
        else:
            single_effects[v] = 0.0

    # Calculate interaction matrix
    n = len(variants)
    interaction = np.zeros((n, n))

    for i, v1 in enumerate(variants):
        for j, v2 in enumerate(variants):
            if i == j:
                interaction[i, j] = single_effects.get(v1, 0)
            else:
                both = [a.actual_hearts or 0 for a in results.analyses if v1 in a.variants and v2 in a.variants]
                if both:
                    actual = np.mean(both)
                    expected = baseline + single_effects.get(v1, 0) + single_effects.get(v2, 0)
                    interaction[i, j] = actual - expected

    return InteractionAnalysis(
        variants=variants,
        single_effects=single_effects,
        interaction_matrix=interaction,
    )


# =============================================================================
# Visualization
# =============================================================================


def plot_comparison(results: ComparisonResults, output_path: str = "difficulty_analysis.png"):
    """Generate comparison dashboard."""
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(14, 10))

    # Filter valid data
    valid = [a for a in results.analyses if a.actual_hearts is not None]
    estimated = [a.estimated_hearts_1k for a in valid]
    actual = [a.actual_hearts for a in valid]

    # 1. Scatter plot
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.scatter(estimated, actual, c="#4ECDC4", alpha=0.6, s=80, edgecolors="white")
    max_val = max(max(estimated), max(actual)) if estimated and actual else 100
    ax1.plot([0, max_val], [0, max_val], "k--", alpha=0.3, label="Perfect")
    ax1.set_xlabel("Estimated Hearts (1k steps)")
    ax1.set_ylabel("Actual Hearts")
    ax1.set_title(f"Prediction Accuracy (r={results.correlation:.3f})")
    ax1.legend()

    # 2. Ratio distribution
    ax2 = fig.add_subplot(2, 2, 2)
    ratios = [a / e for e, a in zip(estimated, actual) if e > 0 and a > 0]
    if ratios:
        ax2.hist(ratios, bins=15, color="#4ECDC4", edgecolor="white", alpha=0.8)
        ax2.axvline(x=1.0, color="red", linestyle="--", label="Perfect")
        ax2.axvline(x=np.median(ratios), color="blue", linestyle="--", label=f"Median={np.median(ratios):.2f}")
        ax2.set_xlabel("Actual / Estimated Ratio")
        ax2.set_ylabel("Count")
        ax2.set_title("Ratio Distribution")
        ax2.legend()

    # 3. Top performers
    ax3 = fig.add_subplot(2, 2, 3)
    sorted_analyses = sorted(valid, key=lambda x: -(x.actual_hearts or 0))[:12]
    names = ["+".join(a.variants[:2])[:18] or "base" for a in sorted_analyses]
    est_vals = [a.estimated_hearts_1k for a in sorted_analyses]
    act_vals = [a.actual_hearts or 0 for a in sorted_analyses]

    x = np.arange(len(names))
    width = 0.35
    ax3.barh(x - width/2, est_vals, width, label="Estimated", color="#3498DB", alpha=0.8)
    ax3.barh(x + width/2, act_vals, width, label="Actual", color="#E74C3C", alpha=0.8)
    ax3.set_yticks(x)
    ax3.set_yticklabels(names, fontsize=8)
    ax3.set_xlabel("Hearts")
    ax3.set_title("Top Performers")
    ax3.legend(loc="lower right")
    ax3.invert_yaxis()

    # 4. Variant bias
    ax4 = fig.add_subplot(2, 2, 4)
    interactions = analyze_interactions(results)
    if interactions.single_effects:
        sorted_effects = sorted(interactions.single_effects.items(), key=lambda x: x[1], reverse=True)
        names = [v[0].replace("_", "\n")[:12] for v in sorted_effects]
        vals = [v[1] for v in sorted_effects]
        colors = ["#2ECC71" if v > 0 else "#E74C3C" for v in vals]
        ax4.barh(range(len(names)), vals, color=colors)
        ax4.set_yticks(range(len(names)))
        ax4.set_yticklabels(names, fontsize=8)
        ax4.axvline(x=0, color="black", linewidth=0.5)
        ax4.set_xlabel("Δ Hearts vs Average")
        ax4.set_title("Variant Impact")
        ax4.invert_yaxis()

    fig.suptitle(f"Difficulty Analysis: {len(results.analyses)} Missions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_interactions(interactions: InteractionAnalysis, output_path: str = "variant_interactions.png"):
    """Generate interaction heatmap."""
    import matplotlib.pyplot as plt

    if not interactions.variants:
        print("No variants to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    vmax = np.max(np.abs(interactions.interaction_matrix))
    im = ax.imshow(interactions.interaction_matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    n = len(interactions.variants)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([v.replace("_", "\n")[:10] for v in interactions.variants], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([v.replace("_", "\n")[:10] for v in interactions.variants], fontsize=8)

    for i in range(n):
        for j in range(n):
            val = interactions.interaction_matrix[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", color=color, fontsize=7)

    ax.set_title("Variant Interaction Matrix\n(diagonal=single effect, off-diagonal=synergy/antagonism)")
    plt.colorbar(im, label="Δ Hearts")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


# =============================================================================
# CLI
# =============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Difficulty Analysis Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Estimate command
    est_parser = subparsers.add_parser("estimate", help="Estimate difficulty for a mission")
    est_parser.add_argument("mission", help="Mission name or 'list' to show available")
    est_parser.add_argument("--thinky", action="store_true", help="Also run thinky agents")
    est_parser.add_argument("--steps", type=int, default=1000, help="Max steps")

    # Compare command
    cmp_parser = subparsers.add_parser("compare", help="Compare estimates vs actual")
    cmp_parser.add_argument("-n", "--samples", type=int, default=50, help="Number of samples")
    cmp_parser.add_argument("--steps", type=int, default=1000, help="Max steps")
    cmp_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    cmp_parser.add_argument("--plot", action="store_true", help="Generate plots")
    cmp_parser.add_argument("-o", "--output", default="comparison_results.json", help="Output file")

    # Interactions command
    int_parser = subparsers.add_parser("interactions", help="Analyze variant interactions")
    int_parser.add_argument("-i", "--input", default="comparison_results.json", help="Input file from compare")
    int_parser.add_argument("--plot", action="store_true", help="Generate plot")

    args = parser.parse_args()

    # Suppress noisy logs
    try:
        from cogames.cli.utils import suppress_noisy_logs
        suppress_noisy_logs()
    except ImportError:
        pass

    if args.command == "estimate":
        if args.mission == "list":
            from cogames.cogs_vs_clips.missions import MISSIONS
            print("Available missions:")
            for m in MISSIONS:
                print(f"  {m.name}")
            return

        from cogames.cogs_vs_clips.missions import MISSIONS
        mission = next((m for m in MISSIONS if m.name == args.mission), None)
        if not mission:
            print(f"Mission '{args.mission}' not found. Use 'list' to see available.")
            return

        analysis = analyze_mission(mission, run_thinky=args.thinky, max_steps=args.steps)
        print(analysis.summary())

    elif args.command == "compare":
        print("=" * 60)
        print("DIFFICULTY COMPARISON")
        print("=" * 60)

        results = run_comparison(
            n_samples=args.samples,
            max_steps=args.steps,
            seed=args.seed,
        )

        results.to_json(args.output)
        print(f"\nSaved to: {args.output}")

        print(results.summary())

        if args.plot:
            plot_comparison(results)

    elif args.command == "interactions":
        with open(args.input) as f:
            data = json.load(f)

        results = ComparisonResults(
            analyses=[
                MissionAnalysis(
                    name=d["name"],
                    variants=d["variants"],
                    estimated_difficulty=d.get("est_difficulty", 0),
                    estimated_steps_per_heart=d.get("est_steps_per_heart", 0),
                    estimated_hearts_1k=d.get("est_hearts_1k", 0),
                    actual_hearts=d.get("actual_hearts"),
                    actual_steps=d.get("actual_steps"),
                )
                for d in data
            ]
        )

        interactions = analyze_interactions(results)

        print("=" * 60)
        print("VARIANT INTERACTIONS")
        print("=" * 60)

        print("\nSingle Effects (Δ hearts):")
        for v, effect in sorted(interactions.single_effects.items(), key=lambda x: -x[1]):
            sign = "+" if effect > 0 else ""
            print(f"  {v}: {sign}{effect:.1f}")

        print("\nTop Synergies:")
        for v1, v2, effect in interactions.top_synergies:
            if effect > 0:
                print(f"  {v1} + {v2}: +{effect:.1f}")

        print("\nTop Antagonisms:")
        for v1, v2, effect in interactions.top_antagonisms:
            if effect < 0:
                print(f"  {v1} + {v2}: {effect:.1f}")

        if args.plot:
            plot_interactions(interactions)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

