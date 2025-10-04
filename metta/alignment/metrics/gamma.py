"""
GAMMA: General Alignment Metric for Multi-agent Autonomy.

Combines individual metrics into collective alignment score.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

from metta.alignment.metrics.base import AlignmentMetric
from metta.alignment.metrics.directional_intent import DirectionalIntentMetric
from metta.alignment.metrics.energy_proportionality import EnergyProportionalityMetric
from metta.alignment.metrics.goal_attainment import GoalAttainmentMetric
from metta.alignment.metrics.path_efficiency import PathEfficiencyMetric
from metta.alignment.metrics.time_efficiency import TimeEfficiencyMetric


class IndividualAlignmentMetric(AlignmentMetric):
    """
    Individual Alignment Metric (IAM_i).

    Combines multiple component metrics into a single alignment score:
        IAM_i = (A_i^w_A · D_i^w_D · E_i^w_E · T_i^w_T · Y_i^w_Y)^(1 / Σw)

    Uses geometric mean to prevent any component from dominating.

    Args:
        weights: Dictionary of component weights (default all 1.0)
        scale: Scale parameter for goal attainment (default 1.0)
        tolerance: Tolerance for directional intent (default 0.05)
        baseline_speed: Baseline speed for time efficiency (default 1.0)
        beta: Calibration for energy proportionality (default 1.0)
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        scale: float = 1.0,
        tolerance: float = 0.05,
        baseline_speed: float = 1.0,
        beta: float = 1.0,
    ):
        """Initialize Individual Alignment Metric."""
        super().__init__(name="Individual Alignment Metric")

        # Default weights
        self.weights = weights or {"A": 1.0, "D": 1.0, "E": 1.0, "T": 1.0, "Y": 1.0}

        # Component metrics
        self.goal_attainment = GoalAttainmentMetric(scale=scale)
        self.directional_intent = DirectionalIntentMetric(tolerance=tolerance)
        self.path_efficiency = PathEfficiencyMetric()
        self.time_efficiency = TimeEfficiencyMetric(baseline_speed=baseline_speed)
        self.energy_proportionality = EnergyProportionalityMetric(beta=beta, tolerance=tolerance)

    def compute(
        self,
        positions: npt.NDArray[np.floating[Any]],
        velocities: npt.NDArray[np.floating[Any]],
        task_directions: npt.NDArray[np.floating[Any]],
        dt: float,
        goal: npt.NDArray[np.floating[Any]] | None = None,
        power: npt.NDArray[np.floating[Any]] | None = None,
        **kwargs: Any,
    ) -> float:
        """
        Compute individual alignment metric.

        Args:
            positions: Agent positions over time, shape (T, d)
            velocities: Agent velocities over time, shape (T, d)
            task_directions: Task direction vectors over time, shape (T, d)
            dt: Time step size
            goal: Optional goal position
            power: Optional power measurements, shape (T,)

        Returns:
            IAM_i ∈ [0, 1], higher means better alignment
        """
        # Compute component metrics
        A_i = self.goal_attainment.compute(positions, velocities, task_directions, dt, goal=goal)
        D_i = self.directional_intent.compute(positions, velocities, task_directions, dt)
        E_i = self.path_efficiency.compute(positions, velocities, task_directions, dt)
        T_i = self.time_efficiency.compute(positions, velocities, task_directions, dt, goal=goal)
        Y_i = self.energy_proportionality.compute(positions, velocities, task_directions, dt, power=power)

        # Weighted geometric mean
        w_A = self.weights.get("A", 1.0)
        w_D = self.weights.get("D", 1.0)
        w_E = self.weights.get("E", 1.0)
        w_T = self.weights.get("T", 1.0)
        w_Y = self.weights.get("Y", 1.0)

        total_weight = w_A + w_D + w_E + w_T + w_Y

        if total_weight == 0:
            return 0.0

        # Add small epsilon to prevent log(0)
        eps = 1e-10
        IAM_i = (
            (A_i + eps) ** w_A * (D_i + eps) ** w_D * (E_i + eps) ** w_E * (T_i + eps) ** w_T * (Y_i + eps) ** w_Y
        ) ** (1.0 / total_weight)

        return float(IAM_i)

    def get_components(
        self,
        positions: npt.NDArray[np.floating[Any]],
        velocities: npt.NDArray[np.floating[Any]],
        task_directions: npt.NDArray[np.floating[Any]],
        dt: float,
        goal: npt.NDArray[np.floating[Any]] | None = None,
        power: npt.NDArray[np.floating[Any]] | None = None,
    ) -> dict[str, float]:
        """
        Get individual component scores.

        Returns:
            Dictionary with keys 'A', 'D', 'E', 'T', 'Y', 'IAM'
        """
        A_i = self.goal_attainment.compute(positions, velocities, task_directions, dt, goal=goal)
        D_i = self.directional_intent.compute(positions, velocities, task_directions, dt)
        E_i = self.path_efficiency.compute(positions, velocities, task_directions, dt)
        T_i = self.time_efficiency.compute(positions, velocities, task_directions, dt, goal=goal)
        Y_i = self.energy_proportionality.compute(positions, velocities, task_directions, dt, power=power)
        IAM_i = self.compute(positions, velocities, task_directions, dt, goal=goal, power=power)

        return {"A": A_i, "D": D_i, "E": E_i, "T": T_i, "Y": Y_i, "IAM": IAM_i}


class GAMMAMetric:
    """
    GAMMA: General Alignment Metric for Multi-agent Autonomy.

    Collective alignment metric for swarms:
        GAMMA = HuberMean({IAM_i})
        GAMMA_α = GAMMA · exp(-α · CV({IAM_i}))

    Args:
        alpha: Dispersion penalty factor (default 0.0, no penalty)
        huber_delta: Huber loss parameter for robust mean (default 1.0)
    """

    def __init__(self, alpha: float = 0.0, huber_delta: float = 1.0):
        """Initialize GAMMA metric."""
        self.alpha = alpha
        self.huber_delta = huber_delta
        self.individual_metric = IndividualAlignmentMetric()

    def compute(
        self,
        agent_trajectories: list[dict[str, npt.NDArray[np.floating[Any]]]],
        dt: float,
        goals: list[npt.NDArray[np.floating[Any]] | None] | None = None,
    ) -> dict[str, float]:
        """
        Compute GAMMA for a swarm.

        Args:
            agent_trajectories: List of trajectory dicts with keys:
                - 'positions': shape (T, d)
                - 'velocities': shape (T, d)
                - 'task_directions': shape (T, d)
            dt: Time step size
            goals: Optional list of goal positions per agent

        Returns:
            Dictionary with keys:
                - 'GAMMA': Collective alignment score
                - 'GAMMA_alpha': Dispersion-penalized score
                - 'IAM_mean': Mean individual alignment
                - 'IAM_std': Standard deviation of IAM
                - 'CV': Coefficient of variation
        """
        if len(agent_trajectories) == 0:
            return {
                "GAMMA": 0.0,
                "GAMMA_alpha": 0.0,
                "IAM_mean": 0.0,
                "IAM_std": 0.0,
                "CV": 0.0,
            }

        # Compute IAM for each agent
        IAM_scores = []
        for i, traj in enumerate(agent_trajectories):
            goal = goals[i] if goals is not None else None
            power = traj.get("power", None)  # Optional power data
            IAM_i = self.individual_metric.compute(
                positions=traj["positions"],
                velocities=traj["velocities"],
                task_directions=traj["task_directions"],
                dt=dt,
                goal=goal,
                power=power,
            )
            IAM_scores.append(IAM_i)

        IAM_array = np.array(IAM_scores)

        # Compute GAMMA (Huber mean)
        GAMMA = self._huber_mean(IAM_array)

        # Compute coefficient of variation
        IAM_mean = float(np.mean(IAM_array))
        IAM_std = float(np.std(IAM_array))
        CV = IAM_std / (IAM_mean + 1e-10)

        # Dispersion-penalized GAMMA
        GAMMA_alpha = GAMMA * np.exp(-self.alpha * CV)

        return {
            "GAMMA": float(GAMMA),
            "GAMMA_alpha": float(GAMMA_alpha),
            "IAM_mean": IAM_mean,
            "IAM_std": IAM_std,
            "CV": CV,
        }

    def _huber_mean(self, values: npt.NDArray[np.floating[Any]]) -> float:
        """
        Compute Huber mean (robust to outliers).

        Args:
            values: Array of values

        Returns:
            Huber mean
        """
        if len(values) == 0:
            return 0.0

        # Apply Huber loss
        # For simplicity, use trimmed mean (robust approximation)
        sorted_values = np.sort(values)
        trim_frac = 0.1  # Trim 10% from each end
        n_trim = int(len(values) * trim_frac)
        if n_trim > 0:
            trimmed = sorted_values[n_trim:-n_trim]
        else:
            trimmed = sorted_values

        return float(np.mean(trimmed))
