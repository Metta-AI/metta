"""Unit tests for Protein optimizer - Easy Problem (2D Quadratic).

Tests convergence on a simple quadratic function that is essentially a multivariate Gaussian.
This should converge quickly and reliably.
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


class QuadraticProblem:
    """Easy: 2D Quadratic function (essentially a multivariate Gaussian).

    f(x, y) = (x - 2)^2 + (y + 1)^2
    Global minimum: f(2, -1) = 0
    """

    def __init__(self):
        self.name = "Quadratic2D"
        self.dim = 2
        self.bounds = [(-5, 5), (-5, 5)]
        self.optimum = np.array([2.0, -1.0])
        self.optimum_value = 0.0
        self.tolerance = 0.3  # Easy problem, should converge closely

    def evaluate(self, x: Dict[str, float]) -> float:
        x0, x1 = x["x0"], x["x1"]
        return (x0 - 2.0) ** 2 + (x1 + 1.0) ** 2

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
        # Check parameter convergence
        param_array = np.array([best_params[f"x{i}"] for i in range(self.dim)])
        param_distance = np.linalg.norm(param_array - self.optimum)

        # Check value convergence
        value_distance = abs(best_value - self.optimum_value)

        return param_distance < self.tolerance and value_distance < self.tolerance


def run_optimization(
    problem: QuadraticProblem, acquisition_fn: str, randomize: bool, num_iterations: int = 50, verbose: bool = False
) -> Dict:
    """Run a single optimization experiment."""
    reset_torch_state()

    # Create Protein optimizer
    config = problem.get_protein_config()
    optimizer = Protein(
        sweep_config=config,
        acquisition_fn=acquisition_fn,
        num_random_samples=5,  # A few more random samples to start
        randomize_acquisition=randomize,
        seed_with_search_center=False,
        ucb_beta=2.0,
        suggestions_per_pareto=32,  # Back to reasonable number
        random_suggestions=512,  # More random suggestions
    )

    # Track best result
    best_value = float("inf")
    best_params = None
    all_values = []
    best_values_over_time = []

    # Run optimization
    for i in range(num_iterations):
        # Get suggestion
        suggestion, info = optimizer.suggest(fill=None)

        # Evaluate objective
        value = problem.evaluate(suggestion)
        all_values.append(value)

        # Track best
        if value < best_value:
            best_value = value
            best_params = suggestion.copy()
            if verbose and (i < 10 or i % 10 == 0 or value < best_value):
                print(f"  Iter {i:3d}: New best = {best_value:.6f} at ({suggestion['x0']:.3f}, {suggestion['x1']:.3f})")

        best_values_over_time.append(best_value)

        # Provide feedback to optimizer (Protein handles minimization internally)
        optimizer.observe(suggestion, value, 1.0, is_failure=False)

    if verbose:
        print(f"  Final best: {best_value:.6f} at ({best_params['x0']:.3f}, {best_params['x1']:.3f})")

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
        "iteration_values": best_values_over_time,
    }


def print_results_table(problem: QuadraticProblem, results: List[Dict]):
    """Print a formatted comparison table of results."""
    print(f"\n{'=' * 80}")
    print(f"EASY PROBLEM: {problem.name} (dim={problem.dim})")
    print(f"Target: {problem.optimum} = {problem.optimum_value:.6f}")
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
def test_quadratic_convergence(acquisition_fn, randomize):
    """Test that Protein optimizer converges on the easy quadratic problem."""
    problem = QuadraticProblem()
    result = run_optimization(problem, acquisition_fn, randomize, num_iterations=50)

    # For easy problem, we expect good convergence
    assert result["value_error"] < 1.0, f"{acquisition_fn} (randomize={randomize}) failed to get close to optimum"

    # At least check we're making progress from random
    assert result["best_value"] < 10.0, (
        f"{acquisition_fn} (randomize={randomize}) failed to improve from random initialization"
    )


def test_all_configurations_comparison():
    """Run all configurations and compare results."""
    problem = QuadraticProblem()
    results = []

    print(f"\nTesting {problem.name} optimization...")

    for acq_fn in ["naive", "ei", "ucb"]:
        for randomize in [False, True]:
            print(f"  Running {acq_fn} (randomize={randomize})...")
            result = run_optimization(problem, acq_fn, randomize, num_iterations=50)
            results.append(result)

    print_results_table(problem, results)

    # Check overall convergence rate - expect high rate for easy problem
    converged_count = sum(1 for r in results if r["converged"])
    convergence_rate = converged_count / len(results)

    assert convergence_rate >= 0.5, f"Convergence rate {convergence_rate:.2f} too low for easy problem"

    # At least half should converge for easy problem
    assert converged_count >= 3, f"Only {converged_count}/6 configurations converged for easy problem"


def test_acquisition_functions_differ():
    """Test that different acquisition functions produce different behaviors."""
    problem = QuadraticProblem()

    trajectories = {}
    for acq_fn in ["naive", "ei", "ucb"]:
        reset_torch_state()

        config = problem.get_protein_config()
        optimizer = Protein(
            sweep_config=config,
            acquisition_fn=acq_fn,
            num_random_samples=3,
            randomize_acquisition=False,
            seed_with_search_center=False,
        )

        # Collect trajectory
        trajectory = []
        for _ in range(10):
            suggestion, _ = optimizer.suggest(fill=None)
            trajectory.append([suggestion["x0"], suggestion["x1"]])
            value = problem.evaluate(suggestion)
            optimizer.observe(suggestion, value, 1.0, is_failure=False)

        trajectories[acq_fn] = np.array(trajectory[3:])  # Skip random phase

    # Check that trajectories differ
    naive_ei_dist = np.mean(np.linalg.norm(trajectories["naive"] - trajectories["ei"], axis=1))
    naive_ucb_dist = np.mean(np.linalg.norm(trajectories["naive"] - trajectories["ucb"], axis=1))
    ei_ucb_dist = np.mean(np.linalg.norm(trajectories["ei"] - trajectories["ucb"], axis=1))

    max_dist = max(naive_ei_dist, naive_ucb_dist, ei_ucb_dist)
    assert max_dist > 0.1, "Acquisition functions produced too similar trajectories"


if __name__ == "__main__":
    """Run the easy problem test standalone."""
    problem = QuadraticProblem()
    results = []

    print("\n" + "=" * 80)
    print("PROTEIN OPTIMIZER - EASY PROBLEM TEST")
    print("=" * 80)

    for acq_fn in ["naive", "ei", "ucb"]:
        for randomize in [False, True]:
            print(f"Running {acq_fn} (randomize={randomize})...")
            result = run_optimization(problem, acq_fn, randomize, num_iterations=50, verbose=True)
            results.append(result)

    print_results_table(problem, results)

    # Print convergence summary
    print("\nConvergence Summary (value at different iterations):")
    print("Acq Function | Randomize | Iter 10 | Iter 20 | Iter 30 | Iter 40 | Final")
    print("-" * 75)
    for r in results:
        vals = r.get("iteration_values", [r["best_value"]] * 5)
        print(
            f"{r['acquisition_fn']:12s} | {str(r['randomize']):9s} | "
            f"{vals[min(9, len(vals) - 1)] if len(vals) > 9 else '-':7.4f} | "
            f"{vals[min(19, len(vals) - 1)] if len(vals) > 19 else '-':7.4f} | "
            f"{vals[min(29, len(vals) - 1)] if len(vals) > 29 else '-':7.4f} | "
            f"{vals[min(39, len(vals) - 1)] if len(vals) > 39 else '-':7.4f} | "
            f"{r['best_value']:7.4f}"
        )

    print("\nTest completed!")
