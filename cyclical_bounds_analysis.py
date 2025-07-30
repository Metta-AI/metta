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
    # Converter B produces at: period/2, period/2 + period, period/2 + 2*period, ...

    hearts_A = max_steps // period

    # B produces at period//2 + n*period for n = 0, 1, 2, ...
    # We want count of times <= max_steps: period//2 + n*period <= max_steps
    # So n <= (max_steps - period//2) / period
    # Number of valid n values (0-indexed) is floor(...) + 1
    if max_steps >= period // 2:
        hearts_B = (max_steps - period // 2) // period + 1
    else:
        hearts_B = 0

    return hearts_A + hearts_B


def both_converters_movement_limited(period, max_steps=500, switch_cost=7):
    """Strategy 3: Greedy simulation for accurate movement-limited bound

    Simulates an optimal agent that makes the best decision at each step:
    after collecting a heart, should it wait here or switch to the other converter?
    """
    if period <= 0:
        return 0

    # If switching costs more than the period, never switch
    if switch_cost >= period:
        return single_converter_hearts(period, max_steps)

    # Pre-generate heart schedules for efficiency (avoid recalculating every iteration)
    # A produces at: period, 2*period, 3*period, ...
    hearts_A = [i * period for i in range(1, (max_steps // period) + 2)]

    # B produces at: period//2, period//2 + period, period//2 + 2*period, ...
    hearts_B = [period // 2 + i * period for i in range(0, ((max_steps + period) // period) + 1)]
    hearts_B = [h for h in hearts_B if h > 0]  # Remove any non-positive times

    # Indices to track next available heart at each converter
    idx_A = 0
    idx_B = 0

    # Decide where to get the first heart
    time_to_get_A = hearts_A[0]  # No switch cost, start at A
    time_to_get_B = max(switch_cost, hearts_B[0])  # Must switch to B

    if time_to_get_A <= time_to_get_B:
        if time_to_get_A > max_steps:
            return 0
        time = time_to_get_A
        location = "A"
        idx_A = 1  # Consumed first heart from A
    else:
        if time_to_get_B > max_steps:
            return 0
        time = time_to_get_B
        location = "B"
        idx_B = 1  # Consumed first heart from B

    hearts = 1

    # Main simulation loop: after each heart, make optimal decision for next
    while time < max_steps:
        # Get next available heart times (or infinity if no more hearts)
        next_A = hearts_A[idx_A] if idx_A < len(hearts_A) else float("inf")
        next_B = hearts_B[idx_B] if idx_B < len(hearts_B) else float("inf")

        # Calculate time to get next heart from each location
        if location == "A":
            # Option 1: Stay at A
            time_if_stay = next_A

            # Option 2: Switch to B
            arrival_at_B = time + switch_cost
            time_if_switch = max(arrival_at_B, next_B)

            # Choose the faster option
            if time_if_stay <= time_if_switch:
                if time_if_stay > max_steps:
                    break
                time = time_if_stay
                idx_A += 1  # Consumed this heart from A
                # location remains 'A'
            else:
                if time_if_switch > max_steps:
                    break
                time = time_if_switch
                location = "B"
                idx_B += 1  # Consumed this heart from B

        else:  # location == 'B'
            # Option 1: Stay at B
            time_if_stay = next_B

            # Option 2: Switch to A
            arrival_at_A = time + switch_cost
            time_if_switch = max(arrival_at_A, next_A)

            # Choose the faster option
            if time_if_stay <= time_if_switch:
                if time_if_stay > max_steps:
                    break
                time = time_if_stay
                idx_B += 1  # Consumed this heart from B
                # location remains 'B'
            else:
                if time_if_switch > max_steps:
                    break
                time = time_if_switch
                location = "A"
                idx_A += 1  # Consumed this heart from A

        hearts += 1

    return hearts


def both_converters_switch_every_heart(period, max_steps=500, switch_cost=7):
    """Strategy 4: Naive switching - switch after every single heart collected

    True naive strategy: alternate A->B->A->B, taking NEXT available heart at each converter.
    This creates phase mismatch waiting time penalties.
    """
    if period <= 0:
        return 0

    # Generate heart schedules for each converter
    hearts_A = [i * period for i in range(1, (max_steps // period) + 2)]
    hearts_B = [period // 2 + i * period for i in range(0, ((max_steps + period) // period) + 1)]
    hearts_B = [h for h in hearts_B if h > 0]

    time = 0
    hearts_collected = 0
    current_location = "A"  # Start at A
    idx_A = 0  # Next heart index for A
    idx_B = 0  # Next heart index for B

    while time < max_steps:
        if current_location == "A":
            # Find next heart from A that appears AFTER current time
            while idx_A < len(hearts_A) and hearts_A[idx_A] <= time:
                idx_A += 1

            if idx_A >= len(hearts_A):
                break  # No more hearts at A

            heart_time = hearts_A[idx_A]
            if heart_time > max_steps:
                break  # Heart appears too late

            # Wait for heart (phase mismatch penalty)
            time = heart_time
            hearts_collected += 1
            idx_A += 1

            # Switch to B
            time += switch_cost
            current_location = "B"

        else:  # current_location == "B"
            # Find next heart from B that appears AFTER current time
            while idx_B < len(hearts_B) and hearts_B[idx_B] <= time:
                idx_B += 1

            if idx_B >= len(hearts_B):
                break  # No more hearts at B

            heart_time = hearts_B[idx_B]
            if heart_time > max_steps:
                break  # Heart appears too late

            # Wait for heart (phase mismatch penalty)
            time = heart_time
            if time > max_steps:
                break

            hearts_collected += 1
            idx_B += 1

            # Switch to A
            time += switch_cost
            current_location = "A"

        # Check if we ran out of time after switching
        if time >= max_steps:
            break

    return hearts_collected


def simulate_memorized_policy(learned_period, test_period, max_steps=500, switch_cost=7):
    """Simulate a policy that learned optimal timing for learned_period, applied to test_period"""
    if learned_period <= 0 or test_period <= 0:
        return 0

    # Generate heart schedules for the TEST period
    hearts_A = [i * test_period for i in range(1, (max_steps // test_period) + 2)]
    hearts_B = [test_period // 2 + i * test_period for i in range(0, ((max_steps + test_period) // test_period) + 1)]
    hearts_B = [h for h in hearts_B if h > 0]

    # The policy "remembers" the optimal switching pattern from the LEARNED period
    # For simplicity, assume it learned: "switch every X ticks" where X is optimal for learned_period

    # Calculate optimal switching interval for the learned period
    # This is a simplified model: switch when it's most efficient
    if switch_cost >= learned_period:
        # If switching costs more than period, learned to never switch
        learned_switch_interval = float("inf")
    else:
        # Learn to switch approximately every 2*learned_period (after getting ~2 hearts)
        # This is a heuristic for what an RL agent might learn
        learned_switch_interval = 2 * learned_period

    time = 0
    hearts_collected = 0
    current_location = "A"
    idx_A = 0
    idx_B = 0
    last_switch_time = 0

    while time < max_steps:
        if current_location == "A":
            # Find next heart from A that appears AFTER current time
            while idx_A < len(hearts_A) and hearts_A[idx_A] <= time:
                idx_A += 1

            if idx_A >= len(hearts_A):
                break

            heart_time = hearts_A[idx_A]
            if heart_time > max_steps:
                break

            # Wait for heart
            time = heart_time
            hearts_collected += 1
            idx_A += 1

            # Check if policy says to switch (based on learned timing)
            if learned_switch_interval != float("inf") and (time - last_switch_time) >= learned_switch_interval:
                time += switch_cost
                current_location = "B"
                last_switch_time = time

        else:  # current_location == "B"
            # Find next heart from B that appears AFTER current time
            while idx_B < len(hearts_B) and hearts_B[idx_B] <= time:
                idx_B += 1

            if idx_B >= len(hearts_B):
                break

            heart_time = hearts_B[idx_B]
            if heart_time > max_steps:
                break

            # Wait for heart
            time = heart_time
            if time > max_steps:
                break

            hearts_collected += 1
            idx_B += 1

            # Check if policy says to switch (based on learned timing)
            if learned_switch_interval != float("inf") and (time - last_switch_time) >= learned_switch_interval:
                time += switch_cost
                current_location = "A"
                last_switch_time = time

        if time >= max_steps:
            break

    return hearts_collected


def analyze_generalization():
    """Analyze how well memorized policies generalize across periods"""
    # Test periods matching your simulation configurations
    test_periods = [12, 16, 20, 22, 30, 32, 36, 40, 44, 50, 66, 100, 200]

    # Create generalization matrix
    generalization_matrix = {}
    optimal_performance = {}

    # First, get optimal performance for each period
    for period in test_periods:
        optimal_performance[period] = both_converters_movement_limited(period)

    # Then test each learned policy on all periods
    for learned_period in test_periods:
        generalization_matrix[learned_period] = {}
        for test_period in test_periods:
            if learned_period == test_period:
                # On its own period, assume it performs optimally
                performance = optimal_performance[test_period]
            else:
                # On other periods, use the memorized policy
                performance = simulate_memorized_policy(learned_period, test_period)

            generalization_matrix[learned_period][test_period] = performance

    return generalization_matrix, optimal_performance, test_periods


def print_generalization_analysis():
    """Print the generalization performance matrix"""
    matrix, optimal, periods = analyze_generalization()

    print("Policy Generalization Analysis")
    print("=" * 80)
    print("Rows: Learned Period | Columns: Test Period | Values: Hearts Collected")
    print("=" * 80)

    # Header
    header = "Learned\\Test"
    for period in periods:
        header += f"{period:>8}"
    print(header)
    print("-" * len(header))

    # Matrix rows
    for learned_period in periods:
        row = f"{learned_period:<12}"
        for test_period in periods:
            performance = matrix[learned_period][test_period]
            if learned_period == test_period:
                row += f"{performance:>8}"  # Diagonal (same period)
            else:
                row += f"{performance:>8}"
        print(row)

    print("\nOptimal Performance (for reference):")
    opt_row = "Optimal     "
    for period in periods:
        opt_row += f"{optimal[period]:>8}"
    print(opt_row)

    # Analysis of generalization patterns
    print("\nGeneralization Insights:")

    # Find which learned policy generalizes best
    avg_performance = {}
    for learned_period in periods:
        total_perf = sum(
            matrix[learned_period][test_period] for test_period in periods if test_period != learned_period
        )
        avg_performance[learned_period] = total_perf / (len(periods) - 1)

    best_generalizer = max(avg_performance.keys(), key=lambda x: avg_performance[x])
    worst_generalizer = min(avg_performance.keys(), key=lambda x: avg_performance[x])

    print(
        f"- Best generalizing policy: learned on period {best_generalizer} "
        f"(avg {avg_performance[best_generalizer]:.1f} hearts)"
    )
    print(
        f"- Worst generalizing policy: learned on period {worst_generalizer} "
        f"(avg {avg_performance[worst_generalizer]:.1f} hearts)"
    )

    # Performance degradation analysis
    total_degradation = 0
    count = 0
    for learned_period in periods:
        for test_period in periods:
            if learned_period != test_period:
                actual = matrix[learned_period][test_period]
                optimal_perf = optimal[test_period]
                if optimal_perf > 0:
                    degradation = (optimal_perf - actual) / optimal_perf
                    total_degradation += degradation
                    count += 1

    avg_degradation = total_degradation / count if count > 0 else 0
    print(f"- Average performance degradation when generalizing: {avg_degradation:.1%}")


def analyze_frequencies():
    """Analyze a dense range of frequencies for smooth curves"""
    # Create a dense range from 2 to 200, ensuring all simulation periods are included
    base_range = list(range(2, 21)) + list(range(21, 51, 2)) + list(range(51, 101, 5)) + list(range(105, 201, 10))

    # Add specific simulation periods to ensure they're all covered
    simulation_periods = [12, 16, 20, 22, 30, 32, 36, 40, 44, 50, 66, 100, 200]

    # Combine and remove duplicates, then sort
    periods = sorted(set(base_range + simulation_periods))

    results = {
        "periods": periods,
        "single_converter": [],
        "both_optimal": [],
        "both_movement_limited": [],
        "switch_every_heart": [],
    }

    for period in periods:
        results["single_converter"].append(single_converter_hearts(period))
        results["both_optimal"].append(both_converters_optimal(period))
        results["both_movement_limited"].append(both_converters_movement_limited(period))
        results["switch_every_heart"].append(both_converters_switch_every_heart(period))

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
        label="Single Converter",
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
        label="Greedy Simulation",
        linewidth=2.5,
    )

    plt.plot(
        results["periods"],
        results["switch_every_heart"],
        "orange",
        label="Switch Every Heart",
        linewidth=2.5,
        linestyle="--",
    )

    plt.xlabel("Converter Period (ticks)", fontsize=14)
    plt.ylabel("Hearts Collected (500 steps)", fontsize=14)
    plt.title("Theoretical Bounds on Heart Collection", fontsize=16, pad=20)
    plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)

    plt.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    plt.xlim(0, 210)
    plt.ylim(0, 150)
    plt.xticks(range(0, 210, 10))

    # Clean up the plot for publication quality
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_linewidth(0.5)
    plt.gca().spines["bottom"].set_linewidth(0.5)

    plt.tight_layout()
    return plt


def print_simulation_table():
    """Print analysis table for actual simulation configurations"""
    results = analyze_frequencies()

    # Your actual simulation timing configurations (timing value -> period = 2 * timing)
    simulation_timings = [6, 8, 10, 11, 15, 16, 18, 20, 22, 25, 33, 50, 100]

    print("MettaGrid Cyclical Timing Simulation Analysis")
    print("=" * 70)
    print("(Effective Period = 2 × Timing Value due to converter phase offset)")
    print("=" * 70)

    print(f"{'Config':<12} {'Period':<8} {'Single':<8} {'Optimal':<8} {'Greedy':<8} {'Naive':<8} {'Gap':<8}")
    print("-" * 70)

    for timing in simulation_timings:
        period = 2 * timing  # Your actual period calculation
        config_name = f"timing_{timing}"

        if period in results["periods"]:
            i = results["periods"].index(period)
            single = results["single_converter"][i]
            optimal = results["both_optimal"][i]
            limited = results["both_movement_limited"][i]
            naive = results["switch_every_heart"][i]
            gap = optimal - single

            print(f"{config_name:<12} {period:<8} {single:<8} {optimal:<8} {limited:<8} {naive:<8} {gap:<8}")
        else:
            print(f"{config_name:<12} {period:<8} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8}")

    print("\nSimulation Details:")
    print(f"- Total configurations: {len(simulation_timings)}")
    print(f"- Period range: {2 * min(simulation_timings)} to {2 * max(simulation_timings)} ticks")
    print("- Each config tests both horizontal and vertical hallway orientations")


def print_analysis():
    """Print detailed analysis of the bounds"""
    results = analyze_frequencies()

    print("\n\nGeneral Cyclical Converter Heart Collection Analysis")
    print("=" * 55)

    print(f"{'Period':<8} {'Single':<8} {'Optimal':<8} {'Greedy':<8} {'Naive':<8} {'Gap':<8}")
    print("-" * 55)

    # Show key representative periods for clarity
    key_periods = [2, 3, 5, 7, 10, 12, 15, 20, 25, 33, 50, 75, 100]
    for period in key_periods:
        if period in results["periods"]:
            i = results["periods"].index(period)
            single = results["single_converter"][i]
            optimal = results["both_optimal"][i]
            limited = results["both_movement_limited"][i]
            naive = results["switch_every_heart"][i]
            gap = optimal - single

            print(f"{period:<8} {single:<8} {optimal:<8} {limited:<8} {naive:<8} {gap:<8}")

    print("\nKey Insights:")
    print("- Single converter strategy provides a reliable lower bound")
    print("- 7-action switching cost makes dual-converter strategy impossible at very high frequencies (period ≤ 7)")
    print("- Greedy simulation: optimal agent making step-by-step stay/switch decisions")
    print("- Naive switching: alternates converters after every heart, limited by ~500/7 ≈ 71 hearts")
    print("- At low frequencies (period >> 7), greedy performance approaches the theoretical optimum")
    print("- The performance gap between strategies is largest at medium frequencies")


if __name__ == "__main__":
    print_simulation_table()
    print_analysis()
    fig = create_figure()
    plt.savefig("cyclical_bounds_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()
