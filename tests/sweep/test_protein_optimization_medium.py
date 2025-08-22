"""Unit tests for Protein optimizer - Medium Problem (Branin Function).

Tests convergence on the Branin function, a classic 2D benchmark with multiple global minima.
This is more challenging than the quadratic but still manageable.
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


class BraninProblem:
    """Medium: Branin function - a classic 2D test function with 3 global minima.

    f(x, y) = a(y - b*x^2 + c*x - r)^2 + s(1 - t)cos(x) + s
    where a=1, b=5.1/(4π^2), c=5/π, r=6, s=10, t=1/(8π)

    Global minima:
    - f(-π, 12.275) = 0.397887
    - f(π, 2.275) = 0.397887
    - f(9.42478, 2.475) = 0.397887
    """

    def __init__(self):
        self.name = "Branin"
        self.dim = 2
        self.bounds = [(-5, 10), (0, 15)]
        # Target the middle global minimum
        self.optimum = np.array([np.pi, 2.275])
        self.optimum_value = 0.397887
        self.tolerance = 0.8  # Medium difficulty, more lenient

    def evaluate(self, x: Dict[str, float]) -> float:
        x0, x1 = x["x0"], x["x1"]
        a = 1.0
        b = 5.1 / (4 * np.pi**2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)

        term1 = a * (x1 - b * x0**2 + c * x0 - r) ** 2
        term2 = s * (1 - t) * np.cos(x0)
        return term1 + term2 + s

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
        """Check if optimizer has converged to any of the global minima."""
        # Check value convergence first (easier criterion)
        value_distance = abs(best_value - self.optimum_value)

        # Check if we're near any of the three global minima
        param_array = np.array([best_params[f"x{i}"] for i in range(self.dim)])

        global_minima = [np.array([-np.pi, 12.275]), np.array([np.pi, 2.275]), np.array([9.42478, 2.475])]

        # Find distance to nearest global minimum
        min_param_distance = min(np.linalg.norm(param_array - gm) for gm in global_minima)

        return min_param_distance < self.tolerance and value_distance < self.tolerance


def run_optimization(problem: BraninProblem, acquisition_fn: str, randomize: bool, num_iterations: int = 50) -> Dict:
    """Run a single optimization experiment."""
    reset_torch_state()

    # Create Protein optimizer
    config = problem.get_protein_config()
    optimizer = Protein(
        sweep_config=config,
        acquisition_fn=acquisition_fn,
        num_random_samples=4,  # A few random samples for exploration
        randomize_acquisition=randomize,
        seed_with_search_center=False,
        ucb_beta=2.0,
        suggestions_per_pareto=24,  # Reduced for speed
        random_suggestions=512,  # Reduce random suggestions
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

    # Distance to nearest global minimum
    global_minima = [np.array([-np.pi, 12.275]), np.array([np.pi, 2.275]), np.array([9.42478, 2.475])]
    param_error = min(np.linalg.norm(param_array - gm) for gm in global_minima)
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
    }


def print_results_table(problem: BraninProblem, results: List[Dict]):
    """Print a formatted comparison table of results."""
    print(f"\n{'=' * 80}")
    print(f"MEDIUM PROBLEM: {problem.name} (dim={problem.dim})")
    print(f"Target (one of three): {problem.optimum} = {problem.optimum_value:.6f}")
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
                "✓" if r["converged"] else "✗",
            ]
        )

    headers = ["Acq. Function", "Randomize", "Best Value", "Value Error", "Param Error", "Converged"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Print summary statistics
    converged_count = sum(1 for r in results if r["converged"])
    print(f"\nConvergence Rate: {converged_count}/{len(results)} ({100 * converged_count / len(results):.0f}%)")

    # Best performer
    best_result = min(results, key=lambda x: x["value_error"])
    print(f"Best Performer: {best_result['acquisition_fn']} (randomize={best_result['randomize']})")
    print(f"Best Value Error: {best_result['value_error']:.6f}")

    # Show which global minimum was found
    param_array = np.array([best_result["best_params"][f"x{i}"] for i in range(problem.dim)])
    if np.linalg.norm(param_array - np.array([-np.pi, 12.275])) < 1.0:
        print("Found global minimum at (-π, 12.275)")
    elif np.linalg.norm(param_array - np.array([np.pi, 2.275])) < 1.0:
        print("Found global minimum at (π, 2.275)")
    elif np.linalg.norm(param_array - np.array([9.42478, 2.475])) < 1.0:
        print("Found global minimum at (9.42478, 2.475)")


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
def test_branin_convergence(acquisition_fn, randomize):
    """Test that Protein optimizer makes progress on the Branin problem."""
    problem = BraninProblem()
    result = run_optimization(problem, acquisition_fn, randomize, num_iterations=50)

    # For medium problem, we expect reasonable progress
    assert result["value_error"] < 5.0, f"{acquisition_fn} (randomize={randomize}) failed to make progress"

    # Should at least beat random search significantly
    assert result["best_value"] < 20.0, f"{acquisition_fn} (randomize={randomize}) performed worse than random"


def test_all_configurations_comparison():
    """Run all configurations and compare results."""
    problem = BraninProblem()
    results = []

    print(f"\nTesting {problem.name} optimization...")

    for acq_fn in ["naive", "ei", "ucb"]:
        for randomize in [False, True]:
            print(f"  Running {acq_fn} (randomize={randomize})...")
            result = run_optimization(problem, acq_fn, randomize, num_iterations=50)
            results.append(result)

    print_results_table(problem, results)

    # Check overall convergence rate - expect moderate rate for medium problem
    converged_count = sum(1 for r in results if r["converged"])
    convergence_rate = converged_count / len(results)

    # We expect at least some configurations to converge
    assert convergence_rate >= 0.17, f"Convergence rate {convergence_rate:.2f} too low for medium problem"

    # At least one should find a good solution
    best_value_error = min(r["value_error"] for r in results)
    assert best_value_error < 2.0, f"Best value error {best_value_error:.2f} too high for medium problem"


def test_multimodal_exploration():
    """Test that optimizer can explore multiple modes of the Branin function."""
    problem = BraninProblem()

    # Run multiple independent optimizations with different seeds
    found_minima = []

    for seed in [42, 123, 456]:
        np.random.seed(seed)
        torch.manual_seed(seed)

        config = problem.get_protein_config()
        optimizer = Protein(
            sweep_config=config,
            acquisition_fn="ei",
            num_random_samples=5,
            randomize_acquisition=True,  # Use randomization for diversity
            seed_with_search_center=False,
        )

        best_value = float("inf")
        best_params = None

        for _ in range(20):
            suggestion, _ = optimizer.suggest(fill=None)
            value = problem.evaluate(suggestion)
            if value < best_value:
                best_value = value
                best_params = suggestion.copy()
            optimizer.observe(suggestion, value, 1.0, is_failure=False)

        if best_value < 5.0:  # Near a minimum
            found_minima.append([best_params["x0"], best_params["x1"]])

    # Check that we found at least one good minimum
    assert len(found_minima) > 0, "Failed to find any minima"

    # Calculate diversity of found minima
    if len(found_minima) > 1:
        found_minima = np.array(found_minima)
        avg_distance = np.mean(
            [
                np.linalg.norm(found_minima[i] - found_minima[j])
                for i in range(len(found_minima))
                for j in range(i + 1, len(found_minima))
            ]
        )
        print(f"Average distance between found minima: {avg_distance:.2f}")


if __name__ == "__main__":
    """Run the medium problem test standalone."""
    problem = BraninProblem()
    results = []

    print("\n" + "=" * 80)
    print("PROTEIN OPTIMIZER - MEDIUM PROBLEM TEST")
    print("=" * 80)

    for acq_fn in ["naive", "ei", "ucb"]:
        for randomize in [False, True]:
            print(f"Running {acq_fn} (randomize={randomize})...")
            result = run_optimization(problem, acq_fn, randomize, num_iterations=50)
            results.append(result)

    print_results_table(problem, results)
    print("\nTest completed!")
