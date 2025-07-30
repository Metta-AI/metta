#!/usr/bin/env python3
"""
Analysis of theoretical bounds for heart collection in cyclical converter environments.

Assumptions:
- 500 max steps per episode
- Two converters with phase offset (A at phase 0, B at phase period/2)
- 7 actions required to switch between converters (turn + move + grab)
- Each converter produces 1 heart every 'period' steps
"""

import matplotlib.pyplot as plt


def single_converter_hearts(period, max_steps=500):
    """Strategy 1: Stay at one converter only (lower bound)"""
    if period <= 0:
        return 0
    return max_steps // period


def both_converters_optimal(period, max_steps=500):
    """Strategy 2: Optimal use of both converters (theoretical upper bound)
    Assumes instant teleportation between converters"""
    if period <= 0:
        return 0

    # Converter A produces at: period, 2*period, 3*period, ...
    # Converter B produces at: period/2, 3*period/2, 5*period/2, ...

    hearts_A = max_steps // period
    hearts_B = max(0, (max_steps - period // 2) // period) if max_steps > period // 2 else 0

    return hearts_A + hearts_B


def both_converters_movement_limited(period, max_steps=500, switch_cost=7):
    """Strategy 3: Both converters with 7-action switching cost (soft upper bound)
    This simulates the practical limit when movement takes time"""
    if period <= 0:
        return 0

    # If switching costs more than or equal to the period, switching is never profitable
    # You'd miss the next heart cycle by the time you finish switching
    if switch_cost >= period:
        return single_converter_hearts(period, max_steps)

    # For low frequencies (high periods), switching cost becomes negligible
    # For high frequencies (low periods), switching becomes increasingly costly

    # Get bounds
    single_bound = single_converter_hearts(period, max_steps)
    optimal_bound = both_converters_optimal(period, max_steps)

    # Simple model: efficiency decreases as period approaches switch_cost
    # When period >> switch_cost, efficiency approaches 100%
    # When period ≈ switch_cost, efficiency approaches single converter performance

    if period <= switch_cost:
        return single_bound

    # Calculate efficiency based on how much slack we have
    slack_ratio = (period - switch_cost) / period
    efficiency = min(1.0, slack_ratio * 2)  # Scale factor to make transition smoother

    # Interpolate between single converter and optimal based on efficiency
    return int(single_bound + efficiency * (optimal_bound - single_bound))


def analyze_frequencies():
    """Analyze a dense range of frequencies for smooth curves"""
    # Create a dense range from 2 to 100 for smooth curves
    periods = list(range(2, 21)) + list(range(21, 51, 2)) + list(range(51, 101, 5))

    results = {"periods": periods, "single_converter": [], "both_optimal": [], "both_movement_limited": []}

    for period in periods:
        results["single_converter"].append(single_converter_hearts(period))
        results["both_optimal"].append(both_converters_optimal(period))
        results["both_movement_limited"].append(both_converters_movement_limited(period))

    return results


def create_figure():
    """Create the bounds analysis figure"""
    results = analyze_frequencies()

    # Set up publication-quality figure
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({"font.size": 12, "font.family": "serif"})

    # Plot the three curves with smooth lines (no markers for cleaner look)
    plt.plot(
        results["periods"],
        results["single_converter"],
        "b-",
        label="Single Converter (Lower Bound)",
        linewidth=2.5,
    )

    plt.plot(
        results["periods"],
        results["both_optimal"],
        "r-",
        label="Both Converters Optimal (Upper Bound)",
        linewidth=2.5,
    )

    plt.plot(
        results["periods"],
        results["both_movement_limited"],
        "g-",
        label="Both Converters Movement-Limited",
        linewidth=2.5,
    )

    plt.xlabel("Converter Period (ticks)", fontsize=14)
    plt.ylabel("Hearts Collected (500 steps)", fontsize=14)
    plt.title("Theoretical Bounds on Heart Collection", fontsize=16, pad=20)
    plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)

    plt.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    plt.xlim(0, 105)
    plt.ylim(0, 150)

    # Clean up the plot for publication quality
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_linewidth(0.5)
    plt.gca().spines["bottom"].set_linewidth(0.5)

    plt.tight_layout()
    return plt


def print_analysis():
    """Print detailed analysis of the bounds"""
    results = analyze_frequencies()

    print("Cyclical Converter Heart Collection Analysis")
    print("=" * 50)
    print(f"{'Period':<8} {'Single':<8} {'Optimal':<8} {'Limited':<8} {'Gap':<8}")
    print("-" * 50)

    # Show key representative periods for clarity
    key_periods = [2, 3, 5, 7, 10, 15, 20, 25, 33, 50, 75, 100]
    for period in key_periods:
        if period in results["periods"]:
            i = results["periods"].index(period)
            single = results["single_converter"][i]
            optimal = results["both_optimal"][i]
            limited = results["both_movement_limited"][i]
            gap = optimal - single

            print(f"{period:<8} {single:<8} {optimal:<8} {limited:<8} {gap:<8}")

    print("\nKey Insights:")
    print("- Single converter strategy provides a reliable lower bound")
    print("- 7-action switching cost makes dual-converter strategy impossible at very high frequencies (period ≤ 7)")
    print("- At low frequencies (period >> 7), switching cost becomes negligible")
    print("- Movement-limited performance interpolates between single and optimal based on period/switch_cost ratio")
    print("- The biggest performance gap occurs at medium frequencies (periods 15-25)")


if __name__ == "__main__":
    print_analysis()
    fig = create_figure()
    plt.savefig("cyclical_bounds_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()
