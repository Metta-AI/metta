"""
Experimental validation of GR-inspired coupling matrices for multi-agent geometric RL.

Tests whether principled coupling matrices (inspired by General Relativity) improve
the correlation between geometric conditioning and coordination difficulty compared
to arbitrary coupling choices.
"""

import warnings
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.linalg import eigh

warnings.filterwarnings("ignore")


@dataclass
class MDPResult:
    """Results for a single MDP experiment."""

    mdp_id: int
    coordination_difficulty: float
    conditioning_baseline: float
    conditioning_state_overlap: float
    conditioning_fisher: float
    conditioning_no_coupling: float


class TwoByTwoMDP:
    """Simple 2x2 MDP for multi-agent experiments."""

    def __init__(self, transition_matrix: np.ndarray, rewards: np.ndarray):
        """
        Args:
            transition_matrix: (2, 2, 2) - P(s'|s,a)
            rewards: (2, 2) - R(s,a)
        """
        assert transition_matrix.shape == (2, 2, 2)
        assert rewards.shape == (2, 2)
        self.P = transition_matrix
        self.R = rewards
        self.n_states = 2
        self.n_actions = 2

    def stationary_distribution(self, policy: np.ndarray) -> np.ndarray:
        """Compute stationary distribution for given policy."""
        # Transition matrix under policy: P_π(s'|s) = Σ_a π(a|s) P(s'|s,a)
        P_pi = np.sum(policy[:, :, np.newaxis] * self.P, axis=1)

        # Solve πP = π (left eigenvector with eigenvalue 1)
        eigenvals, eigenvecs = eigh(P_pi.T)
        idx = np.argmax(eigenvals.real)
        stationary = eigenvecs[:, idx].real
        stationary = np.abs(stationary)  # Ensure positive
        return stationary / stationary.sum()

    def value_iteration(self, policy: np.ndarray, gamma: float = 0.95, tol: float = 1e-6) -> np.ndarray:
        """Compute state values under given policy."""
        V = np.zeros(self.n_states)
        for _ in range(1000):  # Max iterations
            V_new = np.zeros(self.n_states)
            for s in range(self.n_states):
                action_values = []
                for a in range(self.n_actions):
                    next_state_value = sum(self.P[s, a, s_next] * V[s_next] for s_next in range(self.n_states))
                    action_values.append(self.R[s, a] + gamma * next_state_value)
                V_new[s] = sum(policy[s, a] * action_values[a] for a in range(self.n_actions))

            if np.max(np.abs(V_new - V)) < tol:
                break
            V = V_new
        return V

    def nash_equilibrium_convergence_time(self, max_iterations: int = 1000) -> float:
        """Measure coordination difficulty as Nash equilibrium convergence time."""
        # Simple fictitious play to find Nash equilibrium

        # Initialize random policies for two agents
        pi1 = np.random.dirichlet([1, 1], size=self.n_states)
        pi2 = np.random.dirichlet([1, 1], size=self.n_states)

        learning_rate = 0.1
        convergence_threshold = 1e-3

        for iteration in range(max_iterations):
            pi1_old, pi2_old = pi1.copy(), pi2.copy()

            # Update policies via best response (simplified)
            for s in range(self.n_states):
                # Agent 1 best response
                action_values_1 = []
                for a1 in range(self.n_actions):
                    value = 0
                    for a2 in range(self.n_actions):
                        for s_next in range(self.n_states):
                            value += pi2[s, a2] * self.P[s, a1, s_next] * self.R[s_next, a1]
                    action_values_1.append(value)

                best_action_1 = np.argmax(action_values_1)
                target_1 = np.zeros(self.n_actions)
                target_1[best_action_1] = 1.0
                pi1[s] = (1 - learning_rate) * pi1[s] + learning_rate * target_1

                # Agent 2 best response
                action_values_2 = []
                for a2 in range(self.n_actions):
                    value = 0
                    for a1 in range(self.n_actions):
                        for s_next in range(self.n_states):
                            value += pi1[s, a1] * self.P[s, a2, s_next] * self.R[s_next, a2]
                    action_values_2.append(value)

                best_action_2 = np.argmax(action_values_2)
                target_2 = np.zeros(self.n_actions)
                target_2[best_action_2] = 1.0
                pi2[s] = (1 - learning_rate) * pi2[s] + learning_rate * target_2

            # Check convergence
            policy_change = (np.max(np.abs(pi1 - pi1_old)) + np.max(np.abs(pi2 - pi2_old))) / 2

            if policy_change < convergence_threshold:
                return iteration + 1

        return max_iterations  # Didn't converge


def generate_diverse_mdps(n_mdps: int = 21, seed: int = 42) -> List[TwoByTwoMDP]:
    """Generate diverse 2x2 MDPs for testing."""
    np.random.seed(seed)
    mdps = []

    for i in range(n_mdps):
        # Random transition matrices (ensure they're valid probability distributions)
        P = np.random.random((2, 2, 2))
        P = P / P.sum(axis=2, keepdims=True)  # Normalize

        # Random rewards
        R = np.random.random((2, 2)) * 10 - 5  # Range [-5, 5]

        mdps.append(TwoByTwoMDP(P, R))

    return mdps


def compute_coupling_matrices(policies: List[np.ndarray], mdp: TwoByTwoMDP) -> Dict[str, np.ndarray]:
    """Compute all four coupling matrix approaches."""
    n_agents = len(policies)

    # Compute stationary distributions
    stationary_dists = []
    for policy in policies:
        rho = mdp.stationary_distribution(policy)
        stationary_dists.append(rho)

    coupling_matrices = {}

    # 1. Baseline (arbitrary mixing)
    alpha = 0.25
    W_baseline = alpha * np.eye(n_agents) + (1 - alpha) * np.ones((n_agents, n_agents))
    coupling_matrices["baseline"] = W_baseline

    # 2. State visitation overlap (cosine similarity)
    W_overlap = np.zeros((n_agents, n_agents))
    for i in range(n_agents):
        for j in range(n_agents):
            if i == j:
                W_overlap[i, j] = 1.0
            else:
                rho_i, rho_j = stationary_dists[i], stationary_dists[j]
                dot_product = np.dot(rho_i, rho_j)
                norm_i = np.linalg.norm(rho_i)
                norm_j = np.linalg.norm(rho_j)
                W_overlap[i, j] = dot_product / (norm_i * norm_j + 1e-8)
    coupling_matrices["state_overlap"] = W_overlap

    # 3. Fisher distance coupling (simplified)
    W_fisher = np.zeros((n_agents, n_agents))
    beta = 1.0  # Tunable parameter
    for i in range(n_agents):
        for j in range(n_agents):
            if i == j:
                W_fisher[i, j] = 1.0
            else:
                # Simplified Fisher distance (KL divergence proxy)
                pi_i_flat = policies[i].flatten() + 1e-8
                pi_j_flat = policies[j].flatten() + 1e-8
                kl_div = np.sum(pi_i_flat * np.log(pi_i_flat / pi_j_flat))
                W_fisher[i, j] = np.exp(-beta * kl_div)
    coupling_matrices["fisher"] = W_fisher

    # 4. No coupling (identity)
    W_identity = np.eye(n_agents)
    coupling_matrices["no_coupling"] = W_identity

    return coupling_matrices


def compute_geometric_conditioning(policies: List[np.ndarray], mdp: TwoByTwoMDP, coupling_matrix: np.ndarray) -> float:
    """Compute geometric conditioning given coupling matrix."""
    n_agents = len(policies)

    # Compute effective stationary distributions
    stationary_dists = []
    for policy in policies:
        rho = mdp.stationary_distribution(policy)
        stationary_dists.append(rho)

    # Compute agent-specific metric tensors
    conditionings = []

    for i in range(n_agents):
        # Effective stationary distribution for agent i
        rho_eff = np.zeros(mdp.n_states)
        for j in range(n_agents):
            rho_eff += coupling_matrix[i, j] * stationary_dists[j]
        rho_eff = rho_eff / rho_eff.sum()

        # Build metric tensor (diagonal Fisher information)
        metric_blocks = []
        for s in range(mdp.n_states):
            # Diagonal Fisher information at state s
            fisher_diag = 1.0 / (policies[i][s] + 1e-8)
            metric_blocks.append(rho_eff[s] * fisher_diag)

        # Combine into full metric tensor
        metric_tensor = np.diag(np.concatenate(metric_blocks))

        # Compute condition number
        eigenvals = np.real(np.linalg.eigvals(metric_tensor))
        eigenvals = eigenvals[eigenvals > 1e-10]  # Remove near-zero eigenvals

        if len(eigenvals) > 1:
            conditioning = np.max(eigenvals) / np.min(eigenvals)
        else:
            conditioning = 1.0

        conditionings.append(conditioning)

    return float(np.mean(conditionings))


def run_experiment() -> List[MDPResult]:
    """Run the complete experimental protocol."""
    print("Generating diverse 2x2 MDPs...")
    mdps = generate_diverse_mdps(21)

    results = []

    for i, mdp in enumerate(mdps):
        print(f"Processing MDP {i + 1}/21...")

        # Generate random policies for two agents
        np.random.seed(i)  # Reproducible per MDP
        policies = [np.random.dirichlet([1, 1], size=mdp.n_states), np.random.dirichlet([1, 1], size=mdp.n_states)]

        # Measure coordination difficulty
        difficulty = mdp.nash_equilibrium_convergence_time()

        # Compute coupling matrices
        coupling_matrices = compute_coupling_matrices(policies, mdp)

        # Compute geometric conditioning for each coupling approach
        conditioning_baseline = compute_geometric_conditioning(policies, mdp, coupling_matrices["baseline"])
        conditioning_overlap = compute_geometric_conditioning(policies, mdp, coupling_matrices["state_overlap"])
        conditioning_fisher = compute_geometric_conditioning(policies, mdp, coupling_matrices["fisher"])
        conditioning_no_coupling = compute_geometric_conditioning(policies, mdp, coupling_matrices["no_coupling"])

        result = MDPResult(
            mdp_id=i,
            coordination_difficulty=difficulty,
            conditioning_baseline=conditioning_baseline,
            conditioning_state_overlap=conditioning_overlap,
            conditioning_fisher=conditioning_fisher,
            conditioning_no_coupling=conditioning_no_coupling,
        )
        results.append(result)

    return results


def analyze_results(results: List[MDPResult]) -> Dict[str, Dict]:
    """Analyze experimental results and compute correlations."""
    difficulties = [r.coordination_difficulty for r in results]

    conditionings = {
        "baseline": [r.conditioning_baseline for r in results],
        "state_overlap": [r.conditioning_state_overlap for r in results],
        "fisher": [r.conditioning_fisher for r in results],
        "no_coupling": [r.conditioning_no_coupling for r in results],
    }

    analysis = {}

    for method, conditioning_values in conditionings.items():
        result = stats.pearsonr(conditioning_values, difficulties)
        r = result[0]
        p_value = result[1]
        r_squared = r**2

        analysis[method] = {"correlation": r, "p_value": p_value, "r_squared": r_squared, "significant": p_value < 0.05}

    return analysis


def plot_results(results: List[MDPResult], analysis: Dict[str, Dict]):
    """Create visualization of experimental results."""
    difficulties = [r.coordination_difficulty for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("GR-Inspired Coupling Matrix Validation", fontsize=16)

    methods = ["baseline", "state_overlap", "fisher", "no_coupling"]
    titles = ["Baseline (Arbitrary)", "State Overlap (GR)", "Fisher Distance (GR)", "No Coupling"]

    for idx, (method, title) in enumerate(zip(methods, titles, strict=False)):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]

        if method == "baseline":
            conditioning_values = [r.conditioning_baseline for r in results]
        elif method == "state_overlap":
            conditioning_values = [r.conditioning_state_overlap for r in results]
        elif method == "fisher":
            conditioning_values = [r.conditioning_fisher for r in results]
        else:  # no_coupling
            conditioning_values = [r.conditioning_no_coupling for r in results]

        # Scatter plot
        ax.scatter(conditioning_values, difficulties, alpha=0.7)

        # Trend line
        z = np.polyfit(conditioning_values, difficulties, 1)
        p = np.poly1d(z)
        ax.plot(conditioning_values, p(conditioning_values), "r--", alpha=0.8)

        # Formatting
        stats_info = analysis[method]
        ax.set_title(f"{title}\nr = {stats_info['correlation']:.3f}, p = {stats_info['p_value']:.3f}")
        ax.set_xlabel("Geometric Conditioning")
        ax.set_ylabel("Coordination Difficulty")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("gr_coupling_results.png", dpi=300, bbox_inches="tight")
    plt.show()


def print_summary(analysis: Dict[str, Dict]):
    """Print experimental summary."""
    print("\n" + "=" * 60)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("=" * 60)

    print(f"{'Method':<20} {'Correlation':<12} {'R²':<8} {'p-value':<10} {'Significant'}")
    print("-" * 60)

    methods_order = ["baseline", "state_overlap", "fisher", "no_coupling"]
    method_names = {
        "baseline": "Baseline (Arbitrary)",
        "state_overlap": "State Overlap (GR)",
        "fisher": "Fisher Distance (GR)",
        "no_coupling": "No Coupling",
    }

    for method in methods_order:
        stats_info = analysis[method]
        significant = "Yes" if stats_info["significant"] else "No"
        print(
            f"{method_names[method]:<20} {stats_info['correlation']:<12.3f} "
            f"{stats_info['r_squared']:<8.3f} {stats_info['p_value']:<10.3f} {significant}"
        )

    print("\n" + "=" * 60)
    print("HYPOTHESIS TESTING")
    print("=" * 60)

    baseline_r = analysis["baseline"]["correlation"]
    overlap_r = analysis["state_overlap"]["correlation"]
    fisher_r = analysis["fisher"]["correlation"]

    print(f"H1: State overlap > Baseline: {overlap_r:.3f} > {baseline_r:.3f} = {overlap_r > baseline_r}")
    print(f"H2: GR approaches show improvement: State={overlap_r:.3f}, Fisher={fisher_r:.3f}")

    # Find best performing method
    best_method = max(analysis.keys(), key=lambda k: analysis[k]["correlation"])
    best_r = analysis[best_method]["correlation"]
    print(f"Best performing method: {method_names[best_method]} (r = {best_r:.3f})")


if __name__ == "__main__":
    print("Starting GR-Inspired Coupling Matrix Experiment")
    print("This will test whether principled coupling matrices improve")
    print("correlation between geometric conditioning and coordination difficulty.\n")

    # Run experiment
    results = run_experiment()

    # Analyze results
    analysis = analyze_results(results)

    # Print summary
    print_summary(analysis)

    # Create plots
    plot_results(results, analysis)

    print("\nExperiment complete! Results saved to 'gr_coupling_results.png'")
    print("Check the correlation improvements and statistical significance.")
