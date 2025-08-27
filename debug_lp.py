#!/usr/bin/env python3
"""Debug script to understand learning progress algorithm behavior."""

from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressAlgorithm, LearningProgressHypers


def debug_learning_progress():
    """Debug the learning progress calculation."""

    # Create learning progress algorithm
    hypers = LearningProgressHypers(
        ema_timescale=0.1,
        pool_size=10,
        sample_size=5,
        max_samples=10,
    )
    algorithm = LearningProgressAlgorithm(num_tasks=3, hypers=hypers)

    print("=== Debug Learning Progress Algorithm ===")

    # Simulate fast learning task (improving quickly)
    fast_scores = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98, 0.99]

    # Simulate consistent performance task
    consistent_scores = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    # Simulate fast forgetting task
    forget_scores = [0.1, 0.5, 0.9, 0.3, 0.1, 0.05, 0.02, 0.01]

    print("\n=== Task 1: Fast Learning ===")
    for i, score in enumerate(fast_scores):
        algorithm.update(0, score)
        print(f"Update {i}: score={score}")
        if i == 0:  # After first update
            print(f"  Outcomes[0]: {algorithm._lp_tracker._outcomes[0]}")
            print(f"  Random baseline: {algorithm._lp_tracker._random_baseline}")

    print("\n=== Task 2: Consistent Performance ===")
    for i, score in enumerate(consistent_scores):
        algorithm.update(1, score)
        print(f"Update {i}: score={score}")

    print("\n=== Task 3: Fast Forgetting ===")
    for i, score in enumerate(forget_scores):
        algorithm.update(2, score)
        print(f"Update {i}: score={score}")

    # Force update
    algorithm._lp_tracker._update()

    print("\n=== Final State ===")
    print(f"Random baseline: {algorithm._lp_tracker._random_baseline}")
    print(f"Task success rates: {algorithm._lp_tracker._task_success_rate}")
    print(f"P_fast: {algorithm._lp_tracker._p_fast}")
    print(f"P_slow: {algorithm._lp_tracker._p_slow}")

    # Calculate learning progress
    lp_scores = algorithm._lp_tracker._learning_progress()
    print(f"Learning progress scores: {lp_scores}")

    # Check individual components
    print("\n=== Learning Progress Components ===")
    for i in range(3):
        fast = algorithm._lp_tracker._p_fast[i]
        slow = algorithm._lp_tracker._p_slow[i]
        fast_reweighted = algorithm._lp_tracker._reweight(algorithm._lp_tracker._p_fast)[i]
        slow_reweighted = algorithm._lp_tracker._reweight(algorithm._lp_tracker._p_slow)[i]
        lp = lp_scores[i]

        print(f"Task {i}:")
        print(f"  Fast EMA: {fast:.4f}")
        print(f"  Slow EMA: {slow:.4f}")
        print(f"  Fast reweighted: {fast_reweighted:.4f}")
        print(f"  Slow reweighted: {slow_reweighted:.4f}")
        print(f"  Learning progress: {lp:.4f}")
        print(f"  |fast - slow|: {abs(fast - slow):.4f}")
        print(f"  |fast_reweighted - slow_reweighted|: {abs(fast_reweighted - slow_reweighted):.4f}")


if __name__ == "__main__":
    debug_learning_progress()
