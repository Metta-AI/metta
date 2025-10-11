"""
GAMMA evaluator for Metta environments.

Computes alignment metrics from collected trajectories.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

from metta.alignment.metrics.gamma import GAMMAMetric, IndividualAlignmentMetric


class GAMMAEvaluator:
    """
    Evaluates GAMMA alignment metrics for Metta episodes.

    Usage:
        evaluator = GAMMAEvaluator(alpha=0.1)

        # After collecting trajectories
        results = evaluator.evaluate(trajectories, dt=0.1, goals=goal_positions)

        # Log to wandb
        wandb.log({
            "alignment/GAMMA": results["GAMMA"],
            "alignment/IAM_mean": results["IAM_mean"],
        })
    """

    def __init__(
        self,
        alpha: float = 0.1,
        scale: float = 1.0,
        tolerance: float = 0.05,
        baseline_speed: float = 1.0,
        beta: float = 1.0,
    ):
        """
        Initialize GAMMA evaluator.

        Args:
            alpha: Dispersion penalty for GAMMA_α
            scale: Scale for goal attainment
            tolerance: Tolerance for directional intent
            baseline_speed: Baseline speed for time efficiency
            beta: Calibration for energy proportionality
        """
        self.gamma_metric = GAMMAMetric(alpha=alpha, huber_delta=1.0)
        self.individual_metric = IndividualAlignmentMetric(
            scale=scale,
            tolerance=tolerance,
            baseline_speed=baseline_speed,
            beta=beta,
        )

    def evaluate(
        self,
        agent_trajectories: list[dict[str, npt.NDArray[np.floating[Any]]]],
        dt: float,
        goals: list[npt.NDArray[np.floating[Any]] | None] | None = None,
    ) -> dict[str, float]:
        """
        Evaluate GAMMA metrics for a swarm episode.

        Args:
            agent_trajectories: List of trajectory dicts per agent with keys:
                - 'positions': shape (T, d)
                - 'velocities': shape (T, d)
                - 'task_directions': shape (T, d)
                - 'power': shape (T,) [optional]
            dt: Time step size
            goals: Optional list of goal positions per agent

        Returns:
            Dictionary with alignment metrics:
                - 'GAMMA': Collective alignment score
                - 'GAMMA_alpha': Dispersion-penalized score
                - 'IAM_mean': Mean individual alignment
                - 'IAM_std': Standard deviation
                - 'CV': Coefficient of variation
                - 'IAM_scores': List of individual IAM scores
        """
        # Compute GAMMA
        results = self.gamma_metric.compute(agent_trajectories, dt, goals=goals)

        # Compute individual IAM scores for detailed analysis
        IAM_scores = []
        for i, traj in enumerate(agent_trajectories):
            goal = goals[i] if goals is not None else None
            power = traj.get("power", None)
            IAM_i = self.individual_metric.compute(
                positions=traj["positions"],
                velocities=traj["velocities"],
                task_directions=traj["task_directions"],
                dt=dt,
                goal=goal,
                power=power,
            )
            IAM_scores.append(IAM_i)

        results["IAM_scores"] = IAM_scores

        return results

    def evaluate_with_components(
        self,
        agent_trajectories: list[dict[str, npt.NDArray[np.floating[Any]]]],
        dt: float,
        goals: list[npt.NDArray[np.floating[Any]] | None] | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate with detailed component breakdown per agent.

        Returns:
            Dictionary with:
                - All GAMMA results
                - 'components': List of component dicts per agent
        """
        # Get GAMMA results
        results = self.evaluate(agent_trajectories, dt, goals)

        # Get component breakdown for each agent
        components = []
        for i, traj in enumerate(agent_trajectories):
            goal = goals[i] if goals is not None else None
            power = traj.get("power", None)
            comp = self.individual_metric.get_components(
                positions=traj["positions"],
                velocities=traj["velocities"],
                task_directions=traj["task_directions"],
                dt=dt,
                goal=goal,
                power=power,
            )
            components.append(comp)

        results["components"] = components

        return results

    def format_for_wandb(
        self,
        results: dict[str, Any],
        prefix: str = "alignment",
    ) -> dict[str, float]:
        """
        Format results for wandb logging.

        Args:
            results: Results from evaluate() or evaluate_with_components()
            prefix: Metric prefix for wandb

        Returns:
            Flattened dictionary ready for wandb.log()
        """
        wandb_dict = {
            f"{prefix}/GAMMA": results["GAMMA"],
            f"{prefix}/GAMMA_alpha": results["GAMMA_alpha"],
            f"{prefix}/IAM_mean": results["IAM_mean"],
            f"{prefix}/IAM_std": results["IAM_std"],
            f"{prefix}/CV": results["CV"],
        }

        # Add per-agent IAM scores
        if "IAM_scores" in results:
            for i, score in enumerate(results["IAM_scores"]):
                wandb_dict[f"{prefix}/agent_{i}/IAM"] = score

        # Add component averages if available
        if "components" in results:
            components = results["components"]
            if len(components) > 0:
                # Average each component across agents
                for key in ["A", "D", "E", "T", "Y"]:
                    values = [comp[key] for comp in components if key in comp]
                    if values:
                        wandb_dict[f"{prefix}/mean_{key}"] = float(np.mean(values))

        return wandb_dict
