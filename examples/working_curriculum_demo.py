#!/usr/bin/env python3
"""
Working Curriculum Regret Analysis Demo

This script demonstrates the curriculum regret analysis framework
by generating comprehensive tables and visualizations.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from metta.eval.curriculum_analysis import (
    CurriculumRegretAnalyzer,
    CurriculumScenarioAnalyzer,
    create_curriculum_metrics,
)

# Set up plotting style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def generate_comprehensive_data():
    """Generate comprehensive test data for multiple scenarios and curricula."""

    # Define scenarios representing different task dependency structures
    scenarios = {
        "Chain_Simple": {
            "description": "Simple linear task dependencies (0->1->2->...)",
            "oracle_efficiency": 150.0,
            "oracle_time": 30,
        },
        "Chain_HighForget": {
            "description": "Linear dependencies with high forgetting rate",
            "oracle_efficiency": 120.0,
            "oracle_time": 40,
        },
        "Tree_Divergent": {
            "description": "Tree structure with branching dependencies",
            "oracle_efficiency": 130.0,
            "oracle_time": 35,
        },
        "InvertedTree_Convergent": {
            "description": "Inverted tree with converging dependencies",
            "oracle_efficiency": 140.0,
            "oracle_time": 32,
        },
        "DAG_Complex": {"description": "Complex directed acyclic graph", "oracle_efficiency": 110.0, "oracle_time": 45},
    }

    # Define curricula with different characteristics
    curricula_configs = {
        "random": {
            "efficiency_multiplier": 0.65,
            "time_multiplier": 1.6,
            "variance_base": 0.12,
            "description": "Random task selection",
        },
        "topological": {
            "efficiency_multiplier": 0.4,
            "time_multiplier": 2.0,
            "variance_base": 0.08,
            "description": "Topological ordering",
        },
        "reverse_topological": {
            "efficiency_multiplier": 0.95,
            "time_multiplier": 1.2,
            "variance_base": 0.05,
            "description": "Reverse topological ordering",
        },
        "learning_progress": {
            "efficiency_multiplier": 0.8,
            "time_multiplier": 1.3,
            "variance_base": 0.09,
            "description": "Learning progress based",
        },
        "prioritize_regressed": {
            "efficiency_multiplier": 0.75,
            "time_multiplier": 1.4,
            "variance_base": 0.10,
            "description": "Prioritize regressed tasks",
        },
        "variance_based": {
            "efficiency_multiplier": 0.85,
            "time_multiplier": 1.25,
            "variance_base": 0.07,
            "description": "Variance-based selection",
        },
    }

    # Generate comprehensive dataset
    all_results = {}

    for scenario_name, scenario_config in scenarios.items():
        scenario_results = {}

        # Add oracle baseline
        scenario_results["oracle"] = create_curriculum_metrics(
            efficiency=scenario_config["oracle_efficiency"],
            time_to_threshold=scenario_config["oracle_time"],
            time_to_first_mastery=scenario_config["oracle_time"] // 6,
            final_perf_variance=0.05,
            task_weights={"task_a": 0.5, "task_b": 0.5},
        )

        # Add curricula with realistic performance variations
        for curriculum_name, curriculum_config in curricula_configs.items():
            # Add some randomness to make results more realistic
            efficiency_noise = np.random.normal(0, 0.05)
            time_noise = np.random.normal(0, 0.1)

            efficiency = (
                scenario_config["oracle_efficiency"]
                * curriculum_config["efficiency_multiplier"]
                * (1 + efficiency_noise)
            )
            time_to_threshold = int(
                scenario_config["oracle_time"] * curriculum_config["time_multiplier"] * (1 + time_noise)
            )

            # Ensure time_to_threshold is reasonable
            time_to_threshold = max(10, min(200, time_to_threshold))

            # Some curricula might fail in certain scenarios
            if np.random.random() < 0.1:  # 10% chance of failure
                time_to_threshold = -1

            scenario_results[curriculum_name] = create_curriculum_metrics(
                efficiency=efficiency,
                time_to_threshold=time_to_threshold,
                time_to_first_mastery=max(2, time_to_threshold // 5),
                final_perf_variance=curriculum_config["variance_base"] + np.random.normal(0, 0.02),
                task_weights={"task_a": 0.5 + np.random.normal(0, 0.1), "task_b": 0.5 - np.random.normal(0, 0.1)},
            )

        all_results[scenario_name] = scenario_results

    return all_results, scenarios, curricula_configs


def generate_tables(comparison_df):
    """Generate comprehensive tables."""

    print("=" * 80)
    print("CURRICULUM REGRET ANALYSIS RESULTS")
    print("=" * 80)

    # 1. Overall Performance Summary
    print("\n1. OVERALL PERFORMANCE SUMMARY")
    print("-" * 50)

    summary_stats = (
        comparison_df[comparison_df["curriculum"] != "oracle"]
        .groupby("curriculum")
        .agg({"efficiency": ["mean", "std"], "efficiency_regret": ["mean", "std"], "time_regret": ["mean", "std"]})
        .round(2)
    )

    print(summary_stats.to_string())

    # 2. Scenario-by-Scenario Analysis
    print("\n2. SCENARIO-BY-SCENARIO ANALYSIS")
    print("-" * 50)

    for scenario in comparison_df["scenario"].unique():
        scenario_data = comparison_df[comparison_df["scenario"] == scenario]

        print(f"\n{scenario}:")
        scenario_summary = scenario_data[["curriculum", "efficiency", "efficiency_regret", "time_regret"]].round(2)
        print(scenario_summary.to_string(index=False))

    # 3. Best and Worst Performers
    print("\n3. BEST AND WORST PERFORMERS")
    print("-" * 50)

    non_oracle_df = comparison_df[comparison_df["curriculum"] != "oracle"]

    print("\nBest Efficiency (Lowest Regret):")
    best_efficiency = non_oracle_df.loc[non_oracle_df["efficiency_regret"].idxmin()]
    print(f"  {best_efficiency['curriculum']} in {best_efficiency['scenario']}: {best_efficiency['efficiency']:.2f}")

    print("\nWorst Efficiency (Highest Regret):")
    worst_efficiency = non_oracle_df.loc[non_oracle_df["efficiency_regret"].idxmax()]
    print(f"  {worst_efficiency['curriculum']} in {worst_efficiency['scenario']}: {worst_efficiency['efficiency']:.2f}")

    print("\nBest Time Performance (Lowest Time Regret):")
    best_time = non_oracle_df.loc[non_oracle_df["time_regret"].idxmin()]
    print(f"  {best_time['curriculum']} in {best_time['scenario']}: {best_time['time_to_threshold']}")

    print("\nWorst Time Performance (Highest Time Regret):")
    worst_time = non_oracle_df.loc[non_oracle_df["time_regret"].idxmax()]
    print(f"  {worst_time['curriculum']} in {worst_time['scenario']}: {worst_time['time_to_threshold']}")


def generate_plots(comparison_df):
    """Generate comprehensive visualizations."""

    # Filter out oracle for plotting
    plot_df = comparison_df[comparison_df["curriculum"] != "oracle"].copy()

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Curriculum Regret Analysis Results", fontsize=16, fontweight="bold")

    # 1. Efficiency Comparison by Scenario
    ax1 = axes[0, 0]
    scenario_efficiency = plot_df.groupby("scenario")["efficiency"].mean().sort_values(ascending=False)
    scenario_efficiency.plot(kind="bar", ax=ax1, color="skyblue")
    ax1.set_title("Average Efficiency by Scenario")
    ax1.set_ylabel("Efficiency")
    ax1.tick_params(axis="x", rotation=45)

    # 2. Efficiency Regret by Curriculum
    ax2 = axes[0, 1]
    curriculum_regret = plot_df.groupby("curriculum")["efficiency_regret"].mean().sort_values()
    curriculum_regret.plot(kind="bar", ax=ax2, color="lightcoral")
    ax2.set_title("Average Efficiency Regret by Curriculum")
    ax2.set_ylabel("Efficiency Regret")
    ax2.tick_params(axis="x", rotation=45)

    # 3. Time Regret by Curriculum
    ax3 = axes[0, 2]
    curriculum_time_regret = plot_df.groupby("curriculum")["time_regret"].mean().sort_values()
    curriculum_time_regret.plot(kind="bar", ax=ax3, color="lightgreen")
    ax3.set_title("Average Time Regret by Curriculum")
    ax3.set_ylabel("Time Regret")
    ax3.tick_params(axis="x", rotation=45)

    # 4. Efficiency vs Time Regret Scatter
    ax4 = axes[1, 0]
    ax4.scatter(
        plot_df["efficiency_regret"], plot_df["time_regret"], c=plot_df["efficiency"], cmap="viridis", alpha=0.7, s=100
    )
    ax4.set_xlabel("Efficiency Regret")
    ax4.set_ylabel("Time Regret")
    ax4.set_title("Efficiency vs Time Regret")

    # Add colorbar
    scatter = ax4.scatter(
        plot_df["efficiency_regret"], plot_df["time_regret"], c=plot_df["efficiency"], cmap="viridis", alpha=0.7, s=100
    )
    plt.colorbar(scatter, ax=ax4, label="Efficiency")

    # 5. Heatmap: Curriculum Performance by Scenario
    ax5 = axes[1, 1]
    pivot_efficiency = plot_df.pivot_table(values="efficiency", index="scenario", columns="curriculum", aggfunc="mean")
    sns.heatmap(pivot_efficiency, annot=True, fmt=".1f", cmap="RdYlGn", ax=ax5)
    ax5.set_title("Efficiency Heatmap")

    # 6. Final Performance Variance
    ax6 = axes[1, 2]
    variance_data = plot_df.groupby("curriculum")["final_perf_variance"].mean().sort_values()
    variance_data.plot(kind="bar", ax=ax6, color="orange")
    ax6.set_title("Final Performance Variance")
    ax6.set_ylabel("Variance")
    ax6.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("curriculum_regret_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Create additional detailed plots
    create_detailed_plots(plot_df)


def create_detailed_plots(plot_df):
    """Create additional detailed visualizations."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Detailed Curriculum Analysis", fontsize=16, fontweight="bold")

    # 1. Box plot of efficiency by curriculum
    ax1 = axes[0, 0]
    plot_df.boxplot(column="efficiency", by="curriculum", ax=ax1)
    ax1.set_title("Efficiency Distribution by Curriculum")
    ax1.set_xlabel("Curriculum")
    ax1.set_ylabel("Efficiency")
    ax1.tick_params(axis="x", rotation=45)

    # 2. Box plot of efficiency regret by curriculum
    ax2 = axes[0, 1]
    plot_df.boxplot(column="efficiency_regret", by="curriculum", ax=ax2)
    ax2.set_title("Efficiency Regret Distribution by Curriculum")
    ax2.set_xlabel("Curriculum")
    ax2.set_ylabel("Efficiency Regret")
    ax2.tick_params(axis="x", rotation=45)

    # 3. Scatter plot: Efficiency vs Time to Threshold
    ax3 = axes[1, 0]
    valid_time = plot_df[plot_df["time_to_threshold"] != -1]
    ax3.scatter(
        valid_time["efficiency"],
        valid_time["time_to_threshold"],
        c=valid_time["efficiency_regret"],
        cmap="plasma",
        alpha=0.7,
        s=100,
    )
    ax3.set_xlabel("Efficiency")
    ax3.set_ylabel("Time to Threshold")
    ax3.set_title("Efficiency vs Time to Threshold")

    # Add colorbar
    scatter = ax3.scatter(
        valid_time["efficiency"],
        valid_time["time_to_threshold"],
        c=valid_time["efficiency_regret"],
        cmap="plasma",
        alpha=0.7,
        s=100,
    )
    plt.colorbar(scatter, ax=ax3, label="Efficiency Regret")

    # 4. Performance ranking heatmap
    ax4 = axes[1, 1]
    # Create ranking matrix (1 = best, higher = worse)
    ranking_data = plot_df.groupby(["scenario", "curriculum"])["efficiency"].mean().unstack()
    ranking_data = ranking_data.rank(ascending=False, axis=1)

    sns.heatmap(ranking_data, annot=True, fmt=".0f", cmap="RdYlGn_r", ax=ax4)
    ax4.set_title("Curriculum Rankings by Scenario (1=Best)")

    plt.tight_layout()
    plt.savefig("curriculum_detailed_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    """Run the comprehensive curriculum regret analysis demo."""

    print("Curriculum Regret Analysis Framework Demo")
    print("Generating comprehensive tables and visualizations...")
    print("=" * 80)

    # Generate comprehensive test data
    results, scenarios, curricula_configs = generate_comprehensive_data()

    # Create analyzer
    analyzer = CurriculumRegretAnalyzer(max_epochs=200)
    scenario_analyzer = CurriculumScenarioAnalyzer(analyzer)

    # Run comprehensive comparison
    comparison_df = scenario_analyzer.run_scenario_comparison(results)

    # Generate tables
    generate_tables(comparison_df)

    # Generate plots
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    generate_plots(comparison_df)

    print("\n" + "=" * 80)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("Generated files:")
    print("- curriculum_regret_analysis.png")
    print("- curriculum_detailed_analysis.png")
    print("=" * 80)


if __name__ == "__main__":
    main()
