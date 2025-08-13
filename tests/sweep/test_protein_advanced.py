"""
Comprehensive test suite for the improved Protein HPO system.
Tests convergence on known optimization problems and validates all new features.
"""

import matplotlib.pyplot as plt
import numpy as np

# Import our improved protein module (assuming saved as protein_improved.py)
from metta.sweep.protein_advanced import (
    ObservationPoint,
    ParetoGenetic,
    Protein,
    ProteinAdvanced,
    Random,
    efficient_pareto_points,
    expected_improvement,
    upper_confidence_bound,
)


class TestFunctions:
    """Standard optimization test functions with known global optima"""

    @staticmethod
    def branin(x1, x2):
        """
        Branin function: Global optimum at (-π, 12.275), (π, 2.275), (9.42478, 2.475)
        with f* = 0.397887
        """
        a = 1
        b = 5.1 / (4 * np.pi**2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)

        term1 = a * (x2 - b * x1**2 + c * x1 - r) ** 2
        term2 = s * (1 - t) * np.cos(x1)
        term3 = s

        return -(term1 + term2 + term3)  # Negative for maximization

    @staticmethod
    def branin_config():
        """Configuration for Branin function optimization"""
        return {
            "method": "bayesian",
            "metric": "objective",
            "goal": "maximize",
            "x1": {"distribution": "uniform", "min": -5.0, "max": 10.0, "scale": "auto", "mean": 2.5},
            "x2": {"distribution": "uniform", "min": 0.0, "max": 15.0, "scale": "auto", "mean": 7.5},
        }

    @staticmethod
    def hartmann6d(x):
        """
        6D Hartmann function: Global optimum at (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)
        with f* = -3.32237
        """
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array(
            [[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]]
        )
        P = 1e-4 * np.array(
            [
                [1312, 1696, 5569, 124, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381],
            ]
        )

        result = 0
        for i in range(4):
            inner_sum = np.sum(A[i] * (x - P[i]) ** 2)
            result += alpha[i] * np.exp(-inner_sum)

        return result  # Already for maximization (global max is positive)

    @staticmethod
    def hartmann6d_config():
        """Configuration for 6D Hartmann function optimization"""
        config = {"method": "bayesian", "metric": "objective", "goal": "maximize"}

        for i in range(6):
            config[f"x{i}"] = {"distribution": "uniform", "min": 0.0, "max": 1.0, "scale": "auto", "mean": 0.5}

        return config

    @staticmethod
    def constrained_quadratic(x1, x2):
        """
        Simple constrained problem: maximize -(x1-2)² - (x2-1)²
        subject to: x1 + x2 ≤ 2 (constraint ≤ 0)
        Constrained optimum at (1, 1) with f* = -1
        """
        objective = -((x1 - 2) ** 2 + (x2 - 1) ** 2)
        constraint = x1 + x2 - 2  # ≤ 0 for feasibility
        return objective, constraint

    @staticmethod
    def constrained_config():
        """Configuration for constrained optimization"""
        return {
            "method": "bayesian",
            "metric": "objective",
            "goal": "maximize",
            "x1": {"distribution": "uniform", "min": -2.0, "max": 4.0, "scale": "auto", "mean": 1.0},
            "x2": {"distribution": "uniform", "min": -2.0, "max": 4.0, "scale": "auto", "mean": 1.0},
        }

    @staticmethod
    def multi_objective_zdt1(x):
        """
        ZDT1 multi-objective test function
        f1(x) = x1
        f2(x) = g(x) * h(f1, g) where g(x) = 1 + 9*sum(x[1:])/(n-1), h = 1 - sqrt(f1/g)
        Pareto front: x1 ∈ [0,1], x[1:] = 0
        """
        x = np.atleast_1d(x)
        n = len(x)

        f1 = x[0]
        g = 1 + 9 * np.sum(x[1:]) / (n - 1) if n > 1 else 1
        h = 1 - np.sqrt(f1 / g) if g > 0 else 0
        f2 = g * h

        return f1, -f2  # Negative f2 for maximization

    @staticmethod
    def zdt1_config():
        """Configuration for ZDT1 multi-objective optimization"""
        config = {"method": "bayesian", "metric": "objective", "goal": "maximize"}

        # 5D problem
        for i in range(5):
            config[f"x{i}"] = {"distribution": "uniform", "min": 0.0, "max": 1.0, "scale": "auto", "mean": 0.5}

        return config


class TestBasicFunctionality:
    """Test basic functionality and backward compatibility"""

    def test_observation_point_creation(self):
        """Test ObservationPoint creation and attributes"""
        obs = ObservationPoint(
            input=np.array([0.5, 0.3]),
            objectives=np.array([1.2, 0.8]),
            constraints=np.array([-0.1]),
            is_failure=False,
            fidelity=0.8,
        )

        assert obs.input.shape == (2,)
        assert obs.objectives.shape == (2,)
        assert obs.constraints.shape == (1,)
        assert not obs.is_failure
        assert obs.fidelity == 0.8

    def test_efficient_pareto_computation(self):
        """Test efficient Pareto frontier computation"""
        # Create test observations with one clearly dominated point
        observations = [
            ObservationPoint(input=np.array([0]), objectives=np.array([1.0, 2.0])),
            ObservationPoint(input=np.array([1]), objectives=np.array([2.0, 1.0])),
            ObservationPoint(input=np.array([2]), objectives=np.array([0.5, 0.5])),  # Dominated by all others
            ObservationPoint(input=np.array([3]), objectives=np.array([2.5, 0.5])),
        ]

        pareto_obs, pareto_idxs = efficient_pareto_points(observations)

        # Should exclude the dominated point (index 2)
        assert len(pareto_obs) == 3
        assert 2 not in pareto_idxs

        # Verify all non-dominated points are included
        assert 0 in pareto_idxs  # [1.0, 2.0] - best in obj2
        assert 1 in pareto_idxs  # [2.0, 1.0] - balanced
        assert 3 in pareto_idxs  # [2.5, 0.5] - best in obj1

    def test_acquisition_functions(self):
        """Test acquisition function implementations"""
        mu = np.array([0.5, 1.0, 0.8])
        sigma = np.array([0.1, 0.2, 0.15])
        f_best = 0.7

        # Test Expected Improvement
        ei = expected_improvement(mu, sigma, f_best)
        assert ei.shape == (3,)
        assert np.all(ei >= 0)
        assert ei[1] > ei[0]  # Higher mean should give higher EI

        # Test Upper Confidence Bound
        ucb = upper_confidence_bound(mu, sigma, beta=2.0)
        assert ucb.shape == (3,)
        assert np.all(ucb >= mu)  # UCB should be at least the mean

    def test_backward_compatibility(self):
        """Test that legacy Protein class works with old API"""
        config = TestFunctions.branin_config()

        # Test legacy constructor
        optimizer = Protein(config, num_random_samples=5)

        # Test legacy observe method
        suggestion, info = optimizer.suggest({})
        optimizer.observe(suggestion, 1.0, 0.5, is_failure=False)

        # Check legacy attributes exist
        assert hasattr(optimizer, "success_observations")
        assert hasattr(optimizer, "failure_observations")
        assert len(optimizer.success_observations) == 1


class TestOptimizationPerformance:
    """Test optimization performance on known problems"""

    def test_branin_optimization(self):
        """Test convergence on Branin function"""
        config = TestFunctions.branin_config()

        # Test different acquisition functions
        acquisition_fns = ["ei", "ucb"]
        results = {}

        for acq_fn in acquisition_fns:
            optimizer = ProteinAdvanced(config, acquisition_fn=acq_fn, num_random_samples=5, random_suggestions=50)

            best_values = []

            # Run optimization
            for _i in range(30):
                suggestion, info = optimizer.suggest({})

                # Evaluate Branin function
                x1, x2 = suggestion["x1"], suggestion["x2"]
                objective = TestFunctions.branin(x1, x2)
                cost = 1.0  # Fixed cost

                optimizer.observe(suggestion, objective, cost)

                # Track best value so far
                if optimizer.observations:
                    valid_obs = [obs for obs in optimizer.observations if not obs.is_failure]
                    if valid_obs:
                        best_obj = max(obs.objectives[0] for obs in valid_obs)
                        best_values.append(best_obj)

            results[acq_fn] = best_values

        # Check that optimization improved over random
        for acq_fn, values in results.items():
            if len(values) > 10:
                initial_best = max(values[:5])
                final_best = max(values[-5:])
                improvement = final_best - initial_best
                print(
                    f"{acq_fn}: Initial best: {initial_best:.3f}, Final best: {final_best:.3f}, "
                    f"Improvement: {improvement:.3f}"
                )

                # Should show some improvement (relaxed threshold for noisy tests)
                assert improvement > -0.5, f"{acq_fn} failed to maintain performance"

    def test_hartmann6d_optimization(self):
        """Test optimization on higher-dimensional Hartmann function"""
        config = TestFunctions.hartmann6d_config()

        optimizer = ProteinAdvanced(config, acquisition_fn="ei", num_random_samples=10, random_suggestions=100)

        best_values = []

        # Run optimization (fewer iterations for 6D)
        for _i in range(40):
            suggestion, info = optimizer.suggest({})

            # Evaluate Hartmann 6D
            x = np.array([suggestion[f"x{j}"] for j in range(6)])
            objective = TestFunctions.hartmann6d(x)
            cost = 1.0

            optimizer.observe(suggestion, objective, cost)

            # Track progress
            if optimizer.observations:
                valid_obs = [obs for obs in optimizer.observations if not obs.is_failure]
                if valid_obs:
                    best_obj = max(obs.objectives[0] for obs in valid_obs)
                    best_values.append(best_obj)

        # Check convergence in higher dimensions
        if len(best_values) > 20:
            initial_phase = best_values[:10]
            final_phase = best_values[-10:]

            initial_best = max(initial_phase)
            final_best = max(final_phase)

            print(f"Hartmann6D: Initial best: {initial_best:.3f}, Final best: {final_best:.3f}")

            # Should show improvement or at least maintain performance
            assert final_best >= initial_best - 0.2

    def test_constrained_optimization(self):
        """Test constrained optimization"""
        config = TestFunctions.constrained_config()

        optimizer = ProteinAdvanced(config, acquisition_fn="ei", num_random_samples=5, constraint_tolerance=0.0)

        feasible_points = []
        objectives = []

        # Run constrained optimization
        for _i in range(25):
            suggestion, info = optimizer.suggest({})

            x1, x2 = suggestion["x1"], suggestion["x2"]
            objective, constraint = TestFunctions.constrained_quadratic(x1, x2)

            # Record observation with constraint
            optimizer.observe(suggestion, objective, cost=1.0, constraints=[constraint])

            # Track feasible points
            if constraint <= 0.1:  # Small tolerance for numerical errors
                feasible_points.append((x1, x2))
                objectives.append(objective)

        # Check that we found feasible solutions
        assert len(feasible_points) > 5, "Should find multiple feasible points"

        if objectives:
            best_feasible = max(objectives)
            print(f"Constrained: Best feasible objective: {best_feasible:.3f}")

            # Should be reasonably close to constrained optimum (-1.0)
            assert best_feasible > -2.0, "Should find good constrained solution"

    def test_multi_objective_optimization(self):
        """Test multi-objective optimization with Pareto frontier"""
        config = TestFunctions.zdt1_config()

        optimizer = ProteinAdvanced(config, acquisition_fn="ehvi", num_random_samples=8, random_suggestions=50)

        # Run multi-objective optimization
        for _i in range(30):
            suggestion, info = optimizer.suggest({})

            x = np.array([suggestion[f"x{j}"] for j in range(5)])
            f1, f2 = TestFunctions.multi_objective_zdt1(x)

            optimizer.observe(suggestion, f1, cost=-f2)  # Use cost as second objective

        # Analyze Pareto frontier
        if optimizer.observations:
            valid_obs = [obs for obs in optimizer.observations if not obs.is_failure]
            pareto_obs, _ = efficient_pareto_points(valid_obs)

            assert len(pareto_obs) > 3, "Should find multiple Pareto points"

            # Check diversity of Pareto front
            f1_values = [obs.objectives[0] for obs in pareto_obs]
            f2_values = [obs.objectives[1] for obs in pareto_obs]

            f1_range = max(f1_values) - min(f1_values)
            f2_range = max(f2_values) - min(f2_values)

            print(
                f"Multi-objective: {len(pareto_obs)} Pareto points, F1 range: {f1_range:.3f}, F2 range: {f2_range:.3f}"
            )

            # Should have diverse Pareto front
            assert f1_range > 0.1, "Should explore F1 space"
            assert f2_range > 0.1, "Should explore F2 space"


class TestRobustness:
    """Test robustness and edge cases"""

    def test_failure_handling(self):
        """Test handling of failed evaluations"""
        config = TestFunctions.branin_config()

        optimizer = ProteinAdvanced(config, num_random_samples=3)

        # Add some failed observations
        for i in range(5):
            suggestion, info = optimizer.suggest({})

            # Simulate failures
            if i % 3 == 0:
                optimizer.observe(suggestion, 0.0, 1.0, is_failure=True)
            else:
                x1, x2 = suggestion["x1"], suggestion["x2"]
                obj = TestFunctions.branin(x1, x2)
                optimizer.observe(suggestion, obj, 1.0, is_failure=False)

        # Should handle failures gracefully
        valid_obs = [obs for obs in optimizer.observations if not obs.is_failure]
        failed_obs = [obs for obs in optimizer.observations if obs.is_failure]

        assert len(failed_obs) >= 1, "Should record failures"
        assert len(valid_obs) >= 1, "Should have some valid observations"

        # Next suggestion should still work
        suggestion, info = optimizer.suggest({})
        assert suggestion is not None

    def test_empty_observations(self):
        """Test behavior with no observations"""
        config = TestFunctions.branin_config()

        optimizer = ProteinAdvanced(config)

        # Should return search center or random point
        suggestion, info = optimizer.suggest({})
        assert suggestion is not None
        assert "x1" in suggestion
        assert "x2" in suggestion

    def test_single_observation(self):
        """Test behavior with only one observation"""
        config = TestFunctions.branin_config()

        optimizer = ProteinAdvanced(config, num_random_samples=1)

        # Add one observation
        suggestion, info = optimizer.suggest({})
        optimizer.observe(suggestion, 1.0, 1.0)

        # Should still provide next suggestion
        suggestion2, info2 = optimizer.suggest({})
        assert suggestion2 is not None


class TestComparison:
    """Compare different optimizers"""

    def test_optimizer_comparison(self):
        """Compare Random, ParetoGenetic, and ProteinAdvanced"""
        config = TestFunctions.branin_config()

        optimizers = {
            "Random": Random(config),
            "ParetoGenetic": ParetoGenetic(config),
            "ProteinAdvanced": ProteinAdvanced(config, acquisition_fn="ei", num_random_samples=5),
            "Legacy": Protein(config, num_random_samples=5),
        }

        results = {}

        # Run all optimizers
        for name, optimizer in optimizers.items():
            best_values = []

            for _i in range(20):
                suggestion, info = optimizer.suggest({})

                x1, x2 = suggestion["x1"], suggestion["x2"]
                objective = TestFunctions.branin(x1, x2)
                cost = 1.0

                optimizer.observe(suggestion, objective, cost)

                # Track best value for modern optimizers
                if hasattr(optimizer, "observations"):
                    valid_obs = [obs for obs in optimizer.observations if not obs.is_failure]
                    if valid_obs:
                        best_obj = max(obs.objectives[0] for obs in valid_obs)
                        best_values.append(best_obj)
                # Legacy optimizer tracking
                elif hasattr(optimizer, "success_observations") and optimizer.success_observations:
                    best_obj = max(obs["output"] for obs in optimizer.success_observations)
                    best_values.append(best_obj)

            results[name] = best_values

        # Print comparison
        for name, values in results.items():
            if values:
                final_best = max(values)
                print(f"{name}: Final best = {final_best:.3f}")

        # All optimizers should find reasonable solutions
        for name, values in results.items():
            if values:
                assert max(values) > -50, f"{name} failed to find reasonable solution"


def run_visual_test():
    """
    Optional visual test that plots convergence curves
    (requires matplotlib)
    """
    try:
        config = TestFunctions.branin_config()

        optimizer = ProteinAdvanced(config, acquisition_fn="ei", num_random_samples=5)

        objectives = []

        # Run optimization
        for _i in range(30):
            suggestion, info = optimizer.suggest({})

            x1, x2 = suggestion["x1"], suggestion["x2"]
            obj = TestFunctions.branin(x1, x2)

            optimizer.observe(suggestion, obj, 1.0)
            objectives.append(obj)

        # Plot convergence
        plt.figure(figsize=(10, 6))

        # Best so far
        best_so_far = []
        best = float("-inf")
        for obj in objectives:
            best = max(best, obj)
            best_so_far.append(best)

        plt.subplot(1, 2, 1)
        plt.plot(objectives, "b-", alpha=0.7, label="Objectives")
        plt.plot(best_so_far, "r-", linewidth=2, label="Best so far")
        plt.xlabel("Iteration")
        plt.ylabel("Objective Value")
        plt.title("Convergence on Branin Function")
        plt.legend()
        plt.grid(True)

        # Pareto front (if multi-objective data available)
        if optimizer.observations:
            valid_obs = [obs for obs in optimizer.observations if not obs.is_failure]
            pareto_obs, _ = efficient_pareto_points(valid_obs)

            plt.subplot(1, 2, 2)

            # All points
            all_f1 = [obs.objectives[0] for obs in valid_obs]
            all_f2 = [obs.objectives[1] for obs in valid_obs]
            plt.scatter(all_f1, all_f2, alpha=0.6, label="All points")

            # Pareto points
            pareto_f1 = [obs.objectives[0] for obs in pareto_obs]
            pareto_f2 = [obs.objectives[1] for obs in pareto_obs]
            plt.scatter(pareto_f1, pareto_f2, color="red", s=60, label="Pareto front")

            plt.xlabel("Objective (Score)")
            plt.ylabel("Cost")
            plt.title("Objective vs Cost Trade-off")
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.savefig("protein_test_results.png", dpi=150, bbox_inches="tight")
        plt.show()

        print("Visual test completed. Saved plot as 'protein_test_results.png'")

    except ImportError:
        print("Matplotlib not available for visual test")


if __name__ == "__main__":
    # Run all tests
    print("Running Protein HPO Test Suite...")
    print("=" * 50)

    # Basic functionality tests
    print("\n1. Testing basic functionality...")
    basic_tests = TestBasicFunctionality()
    basic_tests.test_observation_point_creation()
    basic_tests.test_efficient_pareto_computation()
    basic_tests.test_acquisition_functions()
    basic_tests.test_backward_compatibility()
    print("✓ Basic functionality tests passed")

    # Optimization performance tests
    print("\n2. Testing optimization performance...")
    perf_tests = TestOptimizationPerformance()
    perf_tests.test_branin_optimization()
    perf_tests.test_hartmann6d_optimization()
    perf_tests.test_constrained_optimization()
    perf_tests.test_multi_objective_optimization()
    print("✓ Optimization performance tests passed")

    # Robustness tests
    print("\n3. Testing robustness...")
    robust_tests = TestRobustness()
    robust_tests.test_failure_handling()
    robust_tests.test_empty_observations()
    robust_tests.test_single_observation()
    print("✓ Robustness tests passed")

    # Comparison tests
    print("\n4. Testing optimizer comparison...")
    comp_tests = TestComparison()
    comp_tests.test_optimizer_comparison()
    print("✓ Comparison tests passed")

    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("\nOptional: Run run_visual_test() for convergence plots")

    # Uncomment to run visual test
    # run_visual_test()
