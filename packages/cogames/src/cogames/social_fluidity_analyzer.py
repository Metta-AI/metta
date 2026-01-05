"""Social Fluid Intelligence metrics for classifying agents as Generalists vs Specialists.

This module implements metrics to analyze agent performance across different team compositions,
identifying agents that are robust (Generalists) vs those with high peak performance but
dependent on specific teammates (Specialists).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from mettagrid.simulator.multi_episode.rollout import MultiEpisodeRolloutResult


class SocialFluidityAnalyzer:
    """Analyzer for Social Fluid Intelligence metrics.

    Takes raw evaluation results and computes three key metrics:
    1. Robustness (Downside Deviation) - measures consistency under adverse conditions
    2. Specialization (Upside Potential) - measures peak performance capability
    3. Team Synergy Sensitivity - measures dependency on specific team compositions
    """

    def __init__(
        self,
        rollout_results: list[MultiEpisodeRolloutResult],
        num_policies: int,
        risk_parameter: float = 1.0,
        top_percentile: float = 0.1,
        n_clusters: int = 3,
    ):
        """Initialize the analyzer.

        Args:
            rollout_results: List of MultiEpisodeRolloutResult from evaluation runs
            num_policies: Number of distinct policies being evaluated
            risk_parameter: Lambda parameter for downside deviation (default 1.0)
            top_percentile: Percentile for upside potential calculation (default 0.1 = top 10%)
            n_clusters: Number of clusters for team synergy analysis (default 3)
        """
        self.rollout_results = rollout_results
        self.num_policies = num_policies
        self.risk_parameter = risk_parameter
        self.top_percentile = top_percentile
        self.n_clusters = n_clusters

        # Build DataFrame with all game results
        self.df = self._build_dataframe()

        # Calculate global statistics
        self.global_mean = float(self.df["score"].mean())
        self.global_variance = float(self.df["score"].var())

    def _build_dataframe(self) -> pd.DataFrame:
        """Build a DataFrame from rollout results with agent scores and team compositions."""
        rows = []

        for mission_idx, mission_result in enumerate(self.rollout_results):
            for episode_idx, episode in enumerate(mission_result.episodes):
                # Get team composition as a tuple of policy indices (sorted for consistency)
                team_composition = tuple(sorted(episode.assignments.tolist()))

                # For each agent, record their score and team
                for agent_id, (policy_idx, score) in enumerate(zip(episode.assignments, episode.rewards, strict=True)):
                    rows.append(
                        {
                            "mission_idx": mission_idx,
                            "episode_idx": episode_idx,
                            "agent_id": agent_id,
                            "policy_idx": int(policy_idx),
                            "score": float(score),
                            "team_composition": team_composition,
                        }
                    )

        return pd.DataFrame(rows)

    def calculate_robustness(self, agent_id: int) -> float:
        """Calculate robustness metric (Downside Deviation) for an agent.

        Formula: μ_i - λ * sqrt(1/N * Σ(min(0, s_ij - τ))²)

        Where:
            - μ_i is the agent's mean score
            - λ is the risk parameter
            - s_ij is the agent's score in game j
            - τ is the global mean score
            - N is the number of games

        Higher values indicate more robust performance (Generalist).

        Args:
            agent_id: Policy index to analyze

        Returns:
            Robustness score (higher = more robust)
        """
        agent_games = self.df[self.df["policy_idx"] == agent_id]
        if len(agent_games) == 0:
            return 0.0

        mean_score = agent_games["score"].mean()
        scores = agent_games["score"].values

        # Calculate downside deviation: sqrt of mean of squared negative deviations from global mean
        negative_deviations = np.minimum(0, scores - self.global_mean)
        downside_variance = np.mean(negative_deviations**2)
        downside_deviation = np.sqrt(downside_variance)

        robustness = mean_score - self.risk_parameter * downside_deviation
        return float(robustness)

    def calculate_specialization(self, agent_id: int) -> float:
        """Calculate specialization metric (Upside Potential) for an agent.

        Formula: (Average score of Agent i's top 10% games) - (Agent i's Median score)

        Higher values indicate higher peak performance potential (Specialist).

        Args:
            agent_id: Policy index to analyze

        Returns:
            Specialization score (higher = more specialized/higher peak potential)
        """
        agent_games = self.df[self.df["policy_idx"] == agent_id]
        if len(agent_games) == 0:
            return 0.0

        scores = agent_games["score"].values
        median_score = np.median(scores)

        # Get top percentile games
        n_top = max(1, int(len(scores) * self.top_percentile))
        top_scores = np.partition(scores, -n_top)[-n_top:]
        top_avg = np.mean(top_scores)

        specialization = top_avg - median_score
        return float(specialization)

    def calculate_team_synergy_sensitivity(self, agent_id: int) -> float:
        """Calculate team synergy sensitivity for an agent.

        This metric clusters teams based on their performance patterns across games,
        then measures how much the agent's performance varies between clusters vs within clusters.

        High ratio = Agent's performance strongly depends on team type (Specialist)
        Low ratio = Agent performs consistently regardless of team type (Generalist) or inconsistently due to noise

        Args:
            agent_id: Policy index to analyze

        Returns:
            Team synergy sensitivity ratio (between-cluster variance / total variance)
        """
        agent_games = self.df[self.df["policy_idx"] == agent_id]
        if len(agent_games) < self.n_clusters:
            # Not enough data to cluster
            return 0.0

        # Step A: Cluster teams based on their performance patterns
        # Build team performance vectors: for each unique team, get scores achieved by that team
        team_performance_vectors = defaultdict(list)
        for _, row in agent_games.iterrows():
            team = row["team_composition"]
            team_performance_vectors[team].append(row["score"])

        # Create embeddings: mean score achieved by each team (across all games with that team)
        team_embeddings = {}
        for team, scores in team_performance_vectors.items():
            team_embeddings[team] = np.mean(scores)

        # Convert to array for clustering
        unique_teams = list(team_embeddings.keys())
        if len(unique_teams) < self.n_clusters:
            # Not enough unique teams to cluster
            return 0.0

        embeddings_array = np.array([team_embeddings[team] for team in unique_teams]).reshape(-1, 1)

        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        team_clusters = kmeans.fit_predict(embeddings_array)
        team_to_cluster = {team: cluster for team, cluster in zip(unique_teams, team_clusters, strict=True)}

        # Step B: Calculate variance ratio
        # Map each agent game to its team's cluster
        agent_games = agent_games.copy()
        agent_games["team_cluster"] = agent_games["team_composition"].map(team_to_cluster)

        # Remove any NaN clusters (teams not in our unique set - shouldn't happen but be safe)
        agent_games = agent_games.dropna(subset=["team_cluster"])

        if len(agent_games) == 0:
            return 0.0

        scores = agent_games["score"].values
        clusters = agent_games["team_cluster"].values.astype(int)

        # Total variance
        total_variance = np.var(scores)

        if total_variance == 0:
            return 0.0

        # Between-cluster variance: variance of cluster means
        cluster_means = []
        for cluster_id in range(self.n_clusters):
            cluster_scores = scores[clusters == cluster_id]
            if len(cluster_scores) > 0:
                cluster_means.append(np.mean(cluster_scores))

        if len(cluster_means) < 2:
            return 0.0

        between_cluster_variance = np.var(cluster_means)

        # Ratio of between-cluster to total variance
        ratio = between_cluster_variance / total_variance
        return float(ratio)

    def analyze_all_agents(self) -> pd.DataFrame:
        """Analyze all agents and return a summary DataFrame.

        Returns:
            DataFrame with columns: policy_idx, robustness, specialization, team_synergy_sensitivity
        """
        results = []
        for policy_idx in range(self.num_policies):
            robustness = self.calculate_robustness(policy_idx)
            specialization = self.calculate_specialization(policy_idx)
            team_synergy = self.calculate_team_synergy_sensitivity(policy_idx)

            results.append(
                {
                    "policy_idx": policy_idx,
                    "robustness": robustness,
                    "specialization": specialization,
                    "team_synergy_sensitivity": team_synergy,
                    "mean_score": float(self.df[self.df["policy_idx"] == policy_idx]["score"].mean()),
                    "global_mean": self.global_mean,
                }
            )

        return pd.DataFrame(results)

    def classify_agent(self, agent_id: int, robustness_threshold: Optional[float] = None, specialization_threshold: Optional[float] = None) -> str:
        """Classify an agent as Generalist or Specialist based on metrics.

        Args:
            agent_id: Policy index to classify
            robustness_threshold: Threshold for robustness (default: median of all agents)
            specialization_threshold: Threshold for specialization (default: median of all agents)

        Returns:
            "Generalist", "Specialist", or "Mixed"
        """
        all_metrics = self.analyze_all_agents()

        if robustness_threshold is None:
            robustness_threshold = all_metrics["robustness"].median()
        if specialization_threshold is None:
            specialization_threshold = all_metrics["specialization"].median()

        agent_robustness = self.calculate_robustness(agent_id)
        agent_specialization = self.calculate_specialization(agent_id)

        is_robust = agent_robustness >= robustness_threshold
        is_specialized = agent_specialization >= specialization_threshold

        if is_robust and not is_specialized:
            return "Generalist"
        elif is_specialized and not is_robust:
            return "Specialist"
        else:
            return "Mixed"
