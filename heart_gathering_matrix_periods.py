#!/usr/bin/env python3
"""
Heart Gathering Matrix Analysis - Period-Based Version

Shows how many hearts an agent collects when it has learned to visit
at a specific PERIOD but is tested on a different PERIOD.

This version explicitly uses periods (not timings) to avoid confusion.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calculate_hearts_gathered(learned_period, test_period, max_steps=500, availability_window=None):
    """
    Calculate hearts gathered when agent learned one period but faces another.

    Args:
        learned_period: Period the agent learned to visit at
        test_period: Actual period of heart production
        max_steps: Maximum simulation steps
        availability_window: How long each heart remains available before being destroyed
                           If None, assumes hearts stay forever
    """
    if learned_period <= 0 or test_period <= 0:
        return 0

    # Agent visits at times: learned_period, 2*learned_period, 3*learned_period, ...
    visit_times = [i * learned_period for i in range(1, (max_steps // learned_period) + 1)]

    # Hearts appear at times: test_period, 2*test_period, 3*test_period, ...
    heart_appearance_times = [i * test_period for i in range(1, (max_steps // test_period) + 1)]

    hearts_collected = 0
    collected_hearts = set()  # Track which hearts have been collected

    for visit_time in visit_times:
        if visit_time > max_steps:
            break

        # Check all hearts that have appeared by this visit time
        for heart_idx, heart_time in enumerate(heart_appearance_times):
            if heart_idx in collected_hearts:
                continue  # Already collected this heart

            if heart_time <= visit_time:
                # Check if heart is still available (not destroyed)
                if availability_window is None:
                    # Hearts stay forever
                    hearts_collected += 1
                    collected_hearts.add(heart_idx)
                    break  # Only collect one heart per visit
                else:
                    # Heart is only available from heart_time to heart_time + availability_window
                    heart_expiry_time = heart_time + availability_window
                    if visit_time <= heart_expiry_time:
                        # Heart is still available
                        hearts_collected += 1
                        collected_hearts.add(heart_idx)
                        break  # Only collect one heart per visit

    return hearts_collected


def create_heart_matrix_for_periods(periods, availability_fraction=0.5):
    """
    Create matrix of hearts gathered for specific periods.

    Args:
        periods: List of periods to analyze
        availability_fraction: Fraction of the period that hearts remain available

    Returns:
        Two matrices: one without heart destruction, one with heart destruction
    """
    n = len(periods)
    matrix_no_destroy = np.zeros((n, n))
    matrix_with_destroy = np.zeros((n, n))

    for i, learned_period in enumerate(periods):
        for j, test_period in enumerate(periods):
            # Without destruction (hearts stay forever)
            matrix_no_destroy[i, j] = calculate_hearts_gathered(learned_period, test_period, availability_window=None)

            # With destruction (hearts only available for part of the period)
            availability_window = int(test_period * availability_fraction)
            matrix_with_destroy[i, j] = calculate_hearts_gathered(
                learned_period, test_period, availability_window=availability_window
            )

    return matrix_no_destroy, matrix_with_destroy


def visualize_period_results(matrix_no_destroy, matrix_with_destroy, periods):
    """Create visualizations with periods explicitly shown on axes."""

    # Create normalized matrices (percentage of optimal)
    normalized_no_destroy = np.zeros_like(matrix_no_destroy)
    normalized_with_destroy = np.zeros_like(matrix_with_destroy)

    for j in range(len(periods)):
        # Normalize each column by its diagonal value (optimal performance)
        if matrix_no_destroy[j, j] > 0:
            normalized_no_destroy[:, j] = (matrix_no_destroy[:, j] / matrix_no_destroy[j, j]) * 100
        if matrix_with_destroy[j, j] > 0:
            normalized_with_destroy[:, j] = (matrix_with_destroy[:, j] / matrix_with_destroy[j, j]) * 100

    # Figure 1: Normalized heatmap comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Use percentage scale (0-100%)
    vmin, vmax = 0, 100

    # Heatmap 1: Without destruction (normalized)
    im1 = ax1.imshow(normalized_no_destroy, cmap="RdBu", aspect="auto", vmin=vmin, vmax=vmax)
    ax1.set_title("Efficiency % - No Destruction", fontsize=14)
    ax1.set_xlabel("Test Period (ticks)", fontsize=12)
    ax1.set_ylabel("Learned Period (ticks)", fontsize=12)

    # Add percentage values to cells
    for i in range(len(periods)):
        for j in range(len(periods)):
            efficiency = normalized_no_destroy[i, j]
            # Use white text for very low and very high values (dark colors)
            text_color = "white" if efficiency < 20 or efficiency > 80 else "black"
            if i == j:
                ax1.text(j, i, "100%", ha="center", va="center", color="white", fontsize=9, weight="bold")
            else:
                ax1.text(j, i, f"{int(efficiency)}%", ha="center", va="center", color=text_color, fontsize=9)

    ax1.set_xticks(range(len(periods)))
    ax1.set_xticklabels(periods, rotation=45)
    ax1.set_yticks(range(len(periods)))
    ax1.set_yticklabels(periods)

    # Heatmap 2: With destruction (normalized)
    im2 = ax2.imshow(normalized_with_destroy, cmap="RdBu", aspect="auto", vmin=vmin, vmax=vmax)
    ax2.set_title("Efficiency % - 50% Availability Window", fontsize=14)
    ax2.set_xlabel("Test Period (ticks)", fontsize=12)
    ax2.set_ylabel("Learned Period (ticks)", fontsize=12)

    # Add percentage values to cells
    for i in range(len(periods)):
        for j in range(len(periods)):
            efficiency = normalized_with_destroy[i, j]
            # Use white text for very low and very high values (dark colors)
            text_color = "white" if efficiency < 20 or efficiency > 80 else "black"
            if i == j:
                ax2.text(j, i, "100%", ha="center", va="center", color="white", fontsize=9, weight="bold")
            else:
                ax2.text(j, i, f"{int(efficiency)}%", ha="center", va="center", color=text_color, fontsize=9)

    ax2.set_xticks(range(len(periods)))
    ax2.set_xticklabels(periods, rotation=45)
    ax2.set_yticks(range(len(periods)))
    ax2.set_yticklabels(periods)

    # Add colorbars
    plt.colorbar(im1, ax=ax1, label="Efficiency (%)")
    plt.colorbar(im2, ax=ax2, label="Efficiency (%)")

    plt.suptitle("Heart Collection Efficiency Matrix (Normalized to Optimal)", fontsize=16)
    plt.tight_layout()
    plt.savefig("heart_matrix_periods_normalized.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Figure 2: Raw values heatmap for reference
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Use consistent color scale for raw values
    vmin_raw = 0
    vmax_raw = max(matrix_no_destroy.max(), matrix_with_destroy.max())

    # Raw heatmap 1: Without destruction
    im1 = ax1.imshow(matrix_no_destroy, cmap="viridis", aspect="auto", vmin=vmin_raw, vmax=vmax_raw)
    ax1.set_title("Hearts Collected (Raw) - No Destruction", fontsize=14)
    ax1.set_xlabel("Test Period (ticks)", fontsize=12)
    ax1.set_ylabel("Learned Period (ticks)", fontsize=12)

    # Add raw values to cells
    for i in range(len(periods)):
        for j in range(len(periods)):
            value = int(matrix_no_destroy[i, j])
            text_color = "white" if value < (vmax_raw / 2) else "black"
            ax1.text(j, i, f"{value}", ha="center", va="center", color=text_color, fontsize=9)

    ax1.set_xticks(range(len(periods)))
    ax1.set_xticklabels(periods, rotation=45)
    ax1.set_yticks(range(len(periods)))
    ax1.set_yticklabels(periods)

    # Raw heatmap 2: With destruction
    im2 = ax2.imshow(matrix_with_destroy, cmap="viridis", aspect="auto", vmin=vmin_raw, vmax=vmax_raw)
    ax2.set_title("Hearts Collected (Raw) - 50% Availability", fontsize=14)
    ax2.set_xlabel("Test Period (ticks)", fontsize=12)
    ax2.set_ylabel("Learned Period (ticks)", fontsize=12)

    # Add raw values to cells
    for i in range(len(periods)):
        for j in range(len(periods)):
            value = int(matrix_with_destroy[i, j])
            text_color = "white" if value < (vmax_raw / 2) else "black"
            ax2.text(j, i, f"{value}", ha="center", va="center", color=text_color, fontsize=9)

    ax2.set_xticks(range(len(periods)))
    ax2.set_xticklabels(periods, rotation=45)
    ax2.set_yticks(range(len(periods)))
    ax2.set_yticklabels(periods)

    # Add colorbars
    plt.colorbar(im1, ax=ax1, label="Hearts Collected")
    plt.colorbar(im2, ax=ax2, label="Hearts Collected")

    plt.suptitle("Heart Collection Raw Values (Period-Based)", fontsize=16)
    plt.tight_layout()
    plt.savefig("heart_matrix_periods_raw.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Figure 3: Normalized performance curves
    fig, ax = plt.subplots(figsize=(10, 6))

    # Select representative periods to plot
    selected_indices = [0, 2, 4, 7, 9, 12]  # periods 12, 20, 30, 40, 50, 200

    for idx in selected_indices:
        if idx < len(periods):
            period = periods[idx]
            # Normalize each row by the diagonal values
            normalized_row = []
            for j in range(len(periods)):
                if matrix_with_destroy[j, j] > 0:
                    normalized_row.append((matrix_with_destroy[idx, j] / matrix_with_destroy[j, j]) * 100)
                else:
                    normalized_row.append(0)

            ax.plot(periods, normalized_row, marker="o", label=f"Learned: {period}", linewidth=2, markersize=5)

    ax.set_xlabel("Test Period (ticks)", fontsize=12)
    ax.set_ylabel("Efficiency (%)", fontsize=12)
    ax.set_title("Collection Efficiency: Learned Period vs Test Period (50% Availability)", fontsize=14)
    ax.legend(title="Learned Period", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")  # Log scale makes period relationships clearer
    ax.set_xticks(periods)
    ax.set_xticklabels(periods, rotation=45)
    ax.set_ylim(0, 105)  # Set y-axis limits for percentages

    # Add horizontal line at 100%
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5, label="Optimal (100%)")

    plt.tight_layout()
    plt.savefig("heart_matrix_period_curves_normalized.png", dpi=300, bbox_inches="tight")
    plt.show()


def print_period_analysis(matrix_no_destroy, matrix_with_destroy, periods, timings):
    """Print analysis using periods explicitly."""

    print("Heart Collection Analysis - Period-Based View")
    print("=" * 80)
    print("Periods analyzed:", periods)
    print("(Corresponding to timings:", timings, ")")
    print("=" * 80)

    # Print the matrix with destruction (most realistic scenario)
    print("\nHearts Collected Matrix (50% Availability Window):")
    print("Rows: Learned Period | Columns: Test Period")
    print("-" * 80)

    # Header
    header = "Learn\\Test"
    for p in periods:
        header += f"{p:>6}"
    print(header)
    print("-" * len(header))

    # Matrix rows
    for i, lp in enumerate(periods):
        row = f"{lp:<10}"
        for j, _tp in enumerate(periods):
            value = int(matrix_with_destroy[i, j])
            if i == j:
                row += f"{value:>6}*"  # Mark diagonal with *
            else:
                row += f"{value:>6}"
        row += f"  | Max possible: {500 // lp}"
        print(row)

    print("\n* = Diagonal (learned period matches test period)")

    # Performance analysis
    print("\n\nKey Insights:")
    print("-" * 80)

    # Find period pairs with perfect synchronization (harmonic relationships)
    print("\nHarmonic Period Pairs (one is multiple of the other):")
    harmonics = []
    for i, p1 in enumerate(periods):
        for j, p2 in enumerate(periods):
            if i != j and (p1 % p2 == 0 or p2 % p1 == 0):
                collected = matrix_with_destroy[i, j]
                optimal = matrix_with_destroy[j, j]
                efficiency = (collected / optimal * 100) if optimal > 0 else 0
                harmonics.append((p1, p2, collected, efficiency))

    harmonics.sort(key=lambda x: x[3], reverse=True)
    for learned, test, hearts, eff in harmonics[:10]:
        print(f"  Learned {learned}, Test {test}: {int(hearts)} hearts ({eff:.0f}% efficiency)")

    # Calculate average performance
    print("\n\nAverage Hearts Collected by Learned Period:")
    for i, period in enumerate(periods):
        avg_hearts = np.mean(matrix_with_destroy[i, :])
        max_possible = 500 // period
        print(f"  Period {period:>3}: {avg_hearts:>5.1f} hearts avg (max possible: {max_possible})")


def main():
    """Generate heart gathering matrix using explicit periods."""

    # Your simulation timings
    simulation_timings = [6, 8, 10, 11, 15, 16, 18, 20, 22, 25, 33, 50, 100]

    # Convert to periods (period = 2 * timing)
    simulation_periods = [2 * t for t in simulation_timings]

    print("Analyzing heart collection using PERIODS (not timings)")
    print(f"Timings: {simulation_timings}")
    print(f"Periods: {simulation_periods}")
    print("Episode length: 500 ticks\n")

    # Show expected hearts for each period
    print("Maximum hearts possible for each period:")
    for timing, period in zip(simulation_timings, simulation_periods, strict=False):
        max_hearts = 500 // period
        print(f"  Timing {timing:>3} → Period {period:>3}: {max_hearts:>2} hearts max")
    print()

    # Generate matrices
    matrix_no_destroy, matrix_with_destroy = create_heart_matrix_for_periods(
        simulation_periods, availability_fraction=0.5
    )

    # Save to CSV with period labels
    df_no_destroy = pd.DataFrame(
        matrix_no_destroy, index=[f"P{p}" for p in simulation_periods], columns=[f"P{p}" for p in simulation_periods]
    )
    df_with_destroy = pd.DataFrame(
        matrix_with_destroy, index=[f"P{p}" for p in simulation_periods], columns=[f"P{p}" for p in simulation_periods]
    )

    # Create normalized dataframes
    normalized_no_destroy = np.zeros_like(matrix_no_destroy)
    normalized_with_destroy = np.zeros_like(matrix_with_destroy)

    for j in range(len(simulation_periods)):
        # Normalize each column by its diagonal value (optimal performance)
        if matrix_no_destroy[j, j] > 0:
            normalized_no_destroy[:, j] = (matrix_no_destroy[:, j] / matrix_no_destroy[j, j]) * 100
        if matrix_with_destroy[j, j] > 0:
            normalized_with_destroy[:, j] = (matrix_with_destroy[:, j] / matrix_with_destroy[j, j]) * 100

    df_normalized_no_destroy = pd.DataFrame(
        normalized_no_destroy,
        index=[f"P{p}" for p in simulation_periods],
        columns=[f"P{p}" for p in simulation_periods],
    )
    df_normalized_with_destroy = pd.DataFrame(
        normalized_with_destroy,
        index=[f"P{p}" for p in simulation_periods],
        columns=[f"P{p}" for p in simulation_periods],
    )

    # Save all CSVs
    df_no_destroy.to_csv("heart_matrix_periods_no_destroy.csv")
    df_with_destroy.to_csv("heart_matrix_periods_with_destroy.csv")
    df_normalized_no_destroy.to_csv("heart_matrix_periods_normalized_no_destroy.csv")
    df_normalized_with_destroy.to_csv("heart_matrix_periods_normalized_with_destroy.csv")
    print("Matrices saved to CSV files (both raw and normalized)\n")

    # Create visualizations
    visualize_period_results(matrix_no_destroy, matrix_with_destroy, simulation_periods)

    # Print analysis
    print_period_analysis(matrix_no_destroy, matrix_with_destroy, simulation_periods, simulation_timings)


if __name__ == "__main__":
    main()
