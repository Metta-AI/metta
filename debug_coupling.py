"""
Diagnostic script to understand why all coupling methods produce identical results.
"""

import numpy as np

from test_gr_coupling import compute_coupling_matrices, compute_geometric_conditioning, generate_diverse_mdps


def debug_coupling_differences():
    """Investigate why coupling methods produce identical results."""

    # Generate a single MDP for testing
    mdps = generate_diverse_mdps(1, seed=42)
    mdp = mdps[0]

    # Generate policies
    np.random.seed(0)
    policies = [np.random.dirichlet([1, 1], size=mdp.n_states), np.random.dirichlet([1, 1], size=mdp.n_states)]

    print("=== DIAGNOSTIC: Coupling Matrix Differences ===")
    print(f"Policy 1:\n{policies[0]}")
    print(f"Policy 2:\n{policies[1]}")

    # Compute coupling matrices
    coupling_matrices = compute_coupling_matrices(policies, mdp)

    print("\n=== COUPLING MATRICES ===")
    for name, matrix in coupling_matrices.items():
        print(f"\n{name}:")
        print(matrix)

    # Compute stationary distributions
    rho1 = mdp.stationary_distribution(policies[0])
    rho2 = mdp.stationary_distribution(policies[1])

    print("\n=== STATIONARY DISTRIBUTIONS ===")
    print(f"Agent 1: {rho1}")
    print(f"Agent 2: {rho2}")
    print(f"Cosine similarity: {np.dot(rho1, rho2) / (np.linalg.norm(rho1) * np.linalg.norm(rho2))}")

    # Compute conditioning for each method
    print("\n=== GEOMETRIC CONDITIONING ===")
    for name, matrix in coupling_matrices.items():
        conditioning = compute_geometric_conditioning(policies, mdp, matrix)
        print(f"{name}: {conditioning}")

    # Test on multiple MDPs to see variance
    print("\n=== TESTING MULTIPLE MDPS ===")
    mdps = generate_diverse_mdps(5, seed=42)

    for method in ["baseline", "state_overlap", "fisher", "no_coupling"]:
        conditionings = []
        for i, mdp in enumerate(mdps):
            np.random.seed(i)
            policies = [np.random.dirichlet([1, 1], size=mdp.n_states), np.random.dirichlet([1, 1], size=mdp.n_states)]
            coupling_matrices = compute_coupling_matrices(policies, mdp)
            conditioning = compute_geometric_conditioning(policies, mdp, coupling_matrices[method])
            conditionings.append(conditioning)

        print(f"{method}: mean={np.mean(conditionings):.6f}, std={np.std(conditionings):.6f}")
        print(f"  Values: {[f'{c:.6f}' for c in conditionings]}")


if __name__ == "__main__":
    debug_coupling_differences()
