"""
Policy Frequency Signature Analysis

This helps identify when a policy has memorized a specific frequency pattern
vs learned a more general strategy.
"""

import matplotlib.pyplot as plt
import numpy as np

from cyclical_bounds_analysis import (
    both_converters_movement_limited,
    both_converters_optimal,
    both_converters_switch_every_heart,
    simulate_memorized_policy,
    single_converter_hearts,
)


def calculate_policy_signatures():
    """Calculate expected performance signatures for different policy types"""

    # Test frequencies (periods)
    test_periods = [12, 16, 20, 22, 30, 32, 36, 40, 44, 50, 66, 100, 200]

    # Calculate bounds for reference
    bounds = {
        "periods": test_periods,
        "lower_bound": [single_converter_hearts(p) for p in test_periods],
        "upper_bound": [both_converters_optimal(p) for p in test_periods],
        "greedy_optimal": [both_converters_movement_limited(p) for p in test_periods],
        "naive_switch": [both_converters_switch_every_heart(p) for p in test_periods],
    }

    # Simulate different "learned" policies
    signatures = {}

    # 1. Single-frequency memorizers
    for memorized_period in [12, 20, 50]:
        signature = []
        for test_period in test_periods:
            if test_period == memorized_period:
                # On trained frequency, assume near-optimal
                score = both_converters_movement_limited(test_period)
            else:
                # On other frequencies, use memorized pattern
                score = simulate_memorized_policy(memorized_period, test_period)
            signature.append(score)
        signatures[f"memorized_{memorized_period}"] = signature

    # 2. Approximate a "generalizer" that learned adaptive switching
    # This would perform reasonably well across frequencies
    generalizer_signature = []
    for period in test_periods:
        # A good generalizer gets ~80-90% of optimal
        optimal = both_converters_movement_limited(period)
        generalizer_score = int(0.85 * optimal)
        generalizer_signature.append(generalizer_score)
    signatures["generalizer"] = generalizer_signature

    # 3. A "stuck" policy (never learned to switch)
    signatures["no_switch"] = bounds["lower_bound"]

    # 4. A "always switch" policy
    signatures["always_switch"] = bounds["naive_switch"]

    return test_periods, bounds, signatures


def calculate_signature_metrics(test_periods, signature, bounds):
    """Calculate metrics that help identify memorization"""

    metrics = {}

    # 1. Performance ratio (how close to optimal)
    optimal = bounds["greedy_optimal"]
    performance_ratios = [s / o if o > 0 else 0 for s, o in zip(signature, optimal, strict=False)]

    # 2. Variance in performance ratio
    metrics["performance_variance"] = np.var(performance_ratios)
    metrics["mean_performance"] = np.mean(performance_ratios)

    # 3. "Spikiness" - how much does performance vary between adjacent frequencies
    differences = [abs(signature[i + 1] - signature[i]) for i in range(len(signature) - 1)]
    metrics["spikiness"] = np.mean(differences)

    # 4. Distance from lower bound
    lower = bounds["lower_bound"]
    above_lower = [s - low for s, low in zip(signature, lower, strict=False)]
    metrics["mean_above_lower"] = np.mean(above_lower)

    return metrics, performance_ratios


def print_signature_analysis():
    """Print analysis to help identify memorization patterns"""

    test_periods, bounds, signatures = calculate_policy_signatures()

    print("POLICY FREQUENCY SIGNATURE ANALYSIS")
    print("=" * 80)
    print("This helps identify when a policy has memorized a specific frequency")
    print("=" * 80)

    # Print performance table
    print(f"\n{'Policy Type':<20}", end="")
    for period in test_periods:
        print(f"{period:>6}", end="")
    print()
    print("-" * (20 + 6 * len(test_periods)))

    # Print bounds first
    print(f"{'Lower Bound':<20}", end="")
    for val in bounds["lower_bound"]:
        print(f"{val:>6}", end="")
    print()

    print(f"{'Upper Bound':<20}", end="")
    for val in bounds["upper_bound"]:
        print(f"{val:>6}", end="")
    print()

    # Print signatures
    for policy_name, signature in signatures.items():
        print(f"{policy_name:<20}", end="")
        for val in signature:
            print(f"{val:>6}", end="")
        print()

    # Plot performance ratios
    plt.figure(figsize=(10, 6))
    for policy_name, signature in signatures.items():
        plt.plot(test_periods, signature, label=policy_name)
    plt.xlabel("Test Period (Heartbeats)")
    plt.ylabel("Performance Score")
    plt.title("Policy Performance vs Test Period")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    print_signature_analysis()
