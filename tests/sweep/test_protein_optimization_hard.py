"""Unit tests for Protein optimizer - Hard Problem (6D Hartmann Function).

Tests convergence on the 6-dimensional Hartmann function, a challenging high-dimensional
benchmark with multiple local minima.
"""

import warnings
from typing import Dict, List

import numpy as np
import pytest
import torch
from tabulate import tabulate

from metta.sweep.protein import Protein

# Suppress PyTorch/Pyro warnings during testing
warnings.filterwarnings("ignore", category=UserWarning)


def reset_torch_state():
    """Reset PyTorch/Pyro state to ensure test idempotency."""
    # Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Reset random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    # Clear Pyro param store
    try:
        import pyro

        pyro.clear_param_store()
    except ImportError:
        pass


class Hartmann6Problem:
    """Hard: 6-dimensional Hartmann function - a challenging high-dimensional problem.

    Has 6 local minima, with one global minimum.
    Global minimum: f(0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573) = -3.32237
    """

    def __init__(self):
        self.name = "Hartmann6D"
        self.dim = 6
        self.bounds = [(0, 1)] * 6
        self.optimum = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])
        self.optimum_value = -3.32237
        self.tolerance = 1.5  # Hard problem, very lenient tolerance

        # Hartmann function parameters
        self.alpha = np.array([1.0, 1.2, 3.0, 3.2])
        self.A = np.array(
            [[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]]
        )
        self.P = 1e-4 * np.array(
            [
                [1312, 1696, 5569, 124, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381],
            ]
        )

    def evaluate(self, x: Dict[str, float]) -> float:
        x_vec = np.array([x[f"x{i}"] for i in range(6)])

        outer_sum = 0
        for i in range(4):
            inner_sum = 0
            for j in range(6):
                inner_sum += self.A[i, j] * (x_vec[j] - self.P[i, j]) ** 2
            outer_sum += self.alpha[i] * np.exp(-inner_sum)

        return -outer_sum

    def get_protein_config(self) -> dict:
        """Get Protein configuration for this problem."""
        config = {
            "metric": self.name,
            "goal": "minimize",
            "method": "bayes",
        }

        # Add parameters with proper bounds
        for i in range(self.dim):
            param_name = f"x{i}"
            config[param_name] = {
                "min": self.bounds[i][0],
                "max": self.bounds[i][1],
                "distribution": "uniform",
                "mean": (self.bounds[i][0] + self.bounds[i][1]) / 2,
                "scale": "auto",
            }

        return config

    def check_convergence(self, best_params: Dict[str, float], best_value: float) -> bool:
        """Check if optimizer has converged to the optimum."""
        # For hard problem, focus mainly on value convergence
        value_distance = abs(best_value - self.optimum_value)

        # Very lenient parameter check
        param_array = np.array([best_params[f"x{i}"] for i in range(self.dim)])
        param_distance = np.linalg.norm(param_array - self.optimum)

        # Accept if we're close in value OR reasonably close in parameters
        return value_distance < 0.5 or (value_distance < 1.0 and param_distance < self.tolerance)


def run_optimization(problem: Hartmann6Problem, acquisition_fn: str, randomize: bool, num_iterations: int = 50) -> Dict:
    """Run a single optimization experiment."""
    reset_torch_state()

    # Create Protein optimizer
    config = problem.get_protein_config()
    optimizer = Protein(
        sweep_config=config,
        acquisition_fn=acquisition_fn,
        num_random_samples=8,  # Some random samples for high-dim exploration
        randomize_acquisition=randomize,
        seed_with_search_center=False,
        ucb_beta=3.0,  # Higher exploration for hard problem
        suggestions_per_pareto=32,  # Reduced for speed
        random_suggestions=512,  # Reduce random suggestions
        expansion_rate=0.3,  # Slightly higher expansion for exploration
    )

    # Track best result
    best_value = float("inf")
    best_params = None
    all_values = []

    # Run optimization
    for _ in range(num_iterations):
        # Get suggestion
        suggestion, info = optimizer.suggest(fill=None)

        # Evaluate objective
        value = problem.evaluate(suggestion)
        all_values.append(value)

        # Track best
        if value < best_value:
            best_value = value
            best_params = suggestion.copy()

        # Provide feedback to optimizer (Protein handles minimization internally)
        optimizer.observe(suggestion, value, 1.0, is_failure=False)

    # Check convergence
    converged = problem.check_convergence(best_params, best_value)

    # Calculate convergence metrics
    param_array = np.array([best_params[f"x{i}"] for i in range(problem.dim)])
    param_error = np.linalg.norm(param_array - problem.optimum)
    value_error = abs(best_value - problem.optimum_value)

    return {
        "acquisition_fn": acquisition_fn,
        "randomize": randomize,
        "best_value": best_value,
        "best_params": best_params,
        "converged": converged,
        "param_error": param_error,
        "value_error": value_error,
        "iterations": num_iterations,
        "improvement": all_values[0] - best_value,  # How much we improved
    }


def print_results_table(problem: Hartmann6Problem, results: List[Dict]):
    """Print a formatted comparison table of results."""
    print(f"\n{'=' * 80}")
    print(f"HARD PROBLEM: {problem.name} (dim={problem.dim})")
    print(f"Target: [...] = {problem.optimum_value:.6f}")
    print(f"Tolerance: {problem.tolerance}")
    print(f"{'=' * 80}")

    # Prepare table data
    table_data = []
    for r in results:
        table_data.append(
            [
                r["acquisition_fn"],
                "Yes" if r["randomize"] else "No",
                f"{r['best_value']:.6f}",
                f"{r['value_error']:.6f}",
                f"{r['param_error']:.4f}",
                f"{r['improvement']:.4f}",
                "✓" if r["converged"] else "✗",
            ]
        )

    headers = ["Acq. Func", "Random", "Best Val", "Val Err", "Param Err", "Improve", "Conv"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Print summary statistics
    converged_count = sum(1 for r in results if r["converged"])
    print(f"\nConvergence Rate: {converged_count}/{len(results)} ({100 * converged_count / len(results):.0f}%)")

    # Best performer
    best_result = min(results, key=lambda x: x["value_error"])
    print(f"Best Performer: {best_result['acquisition_fn']} (randomize={best_result['randomize']})")
    print(f"Best Value: {best_result['best_value']:.6f} (error: {best_result['value_error']:.6f})")

    # Average improvement
    avg_improvement = np.mean([r["improvement"] for r in results])
    print(f"Average Improvement from Random: {avg_improvement:.4f}")


@pytest.mark.parametrize(
    "acquisition_fn,randomize",
    [
        ("naive", False),
        ("naive", True),
        ("ei", False),
        ("ei", True),
        ("ucb", False),
        ("ucb", True),
    ],
)
def test_hartmann_progress(acquisition_fn, randomize):
    """Test that Protein optimizer makes progress on the hard Hartmann problem."""
    problem = Hartmann6Problem()
    result = run_optimization(problem, acquisition_fn, randomize, num_iterations=50)

    # For hard problem, we just expect some improvement
    assert result["improvement"] > 0.5, f"{acquisition_fn} (randomize={randomize}) failed to improve from random"

    # Should at least find something better than -2.0
    assert result["best_value"] < -2.0, f"{acquisition_fn} (randomize={randomize}) failed to find reasonable solution"


def test_all_configurations_comparison():
    """Run all configurations and compare results."""
    problem = Hartmann6Problem()
    results = []

    print(f"\nTesting {problem.name} optimization...")

    for acq_fn in ["naive", "ei", "ucb"]:
        for randomize in [False, True]:
            print(f"  Running {acq_fn} (randomize={randomize})...")
            result = run_optimization(problem, acq_fn, randomize, num_iterations=50)
            results.append(result)

    print_results_table(problem, results)

    # For hard problem, we just want to see improvement
    avg_improvement = np.mean([r["improvement"] for r in results])
    assert avg_improvement > 0.3, f"Average improvement {avg_improvement:.2f} too low"

    # At least one should find a decent solution
    best_value = min(r["best_value"] for r in results)
    assert best_value < -2.5, f"Best value {best_value:.2f} not good enough for hard problem"


def test_high_dimensional_exploration():
    """Test that optimizer can handle high-dimensional space effectively."""
    problem = Hartmann6Problem()

    # Test with extra exploration
    reset_torch_state()

    config = problem.get_protein_config()
    optimizer = Protein(
        sweep_config=config,
        acquisition_fn="ucb",  # UCB tends to explore more
        num_random_samples=15,  # More initial exploration
        randomize_acquisition=True,
        seed_with_search_center=False,
        ucb_beta=4.0,  # High exploration
    )

    values = []
    for _ in range(30):
        suggestion, _ = optimizer.suggest(fill=None)
        value = problem.evaluate(suggestion)
        values.append(value)
        optimizer.observe(suggestion, -value, 1.0, is_failure=False)

    # Check that we're exploring (values should have some variance)
    value_std = np.std(values[:20])  # First 20 includes random phase
    assert value_std > 0.1, "Not exploring enough in high-dimensional space"

    # But also converging (later values should be better)
    early_mean = np.mean(values[:20])
    late_mean = np.mean(values[-20:])
    assert late_mean < early_mean, "Not converging in high-dimensional space"


def test_dimension_scaling():
    """Test that the optimizer handles different problem dimensions."""

    # Create a simpler 3D version for comparison
    class Hartmann3Problem:
        def __init__(self):
            self.name = "Hartmann3D"
            self.dim = 3
            self.bounds = [(0, 1)] * 3
            self.optimum_value = -3.86  # Approximate

        def evaluate(self, x: Dict[str, float]) -> float:
            # Simplified 3D Hartmann
            x_vec = np.array([x[f"x{i}"] for i in range(3)])

            alpha = np.array([1.0, 1.2, 3.0, 3.2])
            A = np.array([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
            P = 1e-4 * np.array([[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]])

            outer_sum = 0
            for i in range(4):
                inner_sum = sum(A[i, j] * (x_vec[j] - P[i, j]) ** 2 for j in range(3))
                outer_sum += alpha[i] * np.exp(-inner_sum)

            return -outer_sum

        def get_protein_config(self) -> dict:
            config = {
                "metric": self.name,
                "goal": "minimize",
                "method": "bayes",
            }
            for i in range(self.dim):
                config[f"x{i}"] = {
                    "min": 0,
                    "max": 1,
                    "distribution": "uniform",
                    "mean": 0.5,
                    "scale": "auto",
                }
            return config

    # Compare 3D vs 6D performance
    problem_3d = Hartmann3Problem()
    problem_6d = Hartmann6Problem()

    results = {}
    for problem in [problem_3d, problem_6d]:
        reset_torch_state()

        config = problem.get_protein_config()
        optimizer = Protein(
            sweep_config=config,
            acquisition_fn="ei",
            num_random_samples=5,
            randomize_acquisition=False,
            seed_with_search_center=False,
        )

        best_value = float("inf")
        for _ in range(20):
            suggestion, _ = optimizer.suggest(fill=None)
            value = problem.evaluate(suggestion)
            best_value = min(best_value, value)
            optimizer.observe(suggestion, value, 1.0, is_failure=False)

        results[problem.name] = best_value

    print(f"3D Hartmann best: {results['Hartmann3D']:.4f}")
    print(f"6D Hartmann best: {results['Hartmann6D']:.4f}")

    # 3D should generally perform better due to lower dimensionality
    # But both should find reasonable solutions
    assert results["Hartmann3D"] < -2.0, "3D Hartmann optimization failed"
    assert results["Hartmann6D"] < -1.5, "6D Hartmann optimization failed"


if __name__ == "__main__":
    """Run the hard problem test standalone."""
    problem = Hartmann6Problem()
    results = []

    print("\n" + "=" * 80)
    print("PROTEIN OPTIMIZER - HARD PROBLEM TEST")
    print("=" * 80)

    for acq_fn in ["naive", "ei", "ucb"]:
        for randomize in [False, True]:
            print(f"Running {acq_fn} (randomize={randomize})...")
            result = run_optimization(problem, acq_fn, randomize, num_iterations=50)
            results.append(result)

    print_results_table(problem, results)
    print("\nTest completed!")
